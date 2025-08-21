# main.py
import config
import google.generativeai as genai
import json
import asyncio
import secrets
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from supabase import AsyncClient
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Literal, Annotated
import uuid
from datetime import datetime, timedelta

# --- Pydantic models ---
class Transaction(BaseModel):
    transaction_id: uuid.UUID
    amount: float = Field(..., example=100.00)
    currency: str = Field(..., example="USD")
    country: str = Field(..., alias='geo', example="US")
    card_bin: Optional[str] = Field(None, example="424242")
    payment_method: Optional[str] = Field(None, example="credit_card")
    transaction_type: Literal['payin', 'payout']

class TransactionStatusUpdate(BaseModel):
    transaction_id: uuid.UUID
    status: str

class RankedPsp(BaseModel):
    rank: int
    psp_id: str
    psp_name: str
    score: float
    reason: str

class RoutingResponse(BaseModel):
    ranked_psps: List[RankedPsp]

class DashboardStats(BaseModel):
    total_volume_24h: float
    total_transactions_24h: int
    success_rate_24h: float
    avg_speed: str

class TransactionDetail(BaseModel):
    id: uuid.UUID
    created_at: datetime
    amount: float
    currency: str
    geo: str
    status: Optional[str] = None

class PaginatedTransactionsResponse(BaseModel):
    transactions: List[TransactionDetail]
    total_count: int
    page: int
    page_size: int

class ApiKeyResponse(BaseModel):
    api_key: Optional[str] = None

class AdminUser(BaseModel):
    id: uuid.UUID
    email: Optional[str] = None
    created_at: datetime
    last_sign_in_at: Optional[datetime] = None

class PspBase(BaseModel):
    name: str
    speed_score: Optional[float] = None
    risk_score: Optional[float] = None
    is_active: Optional[bool] = True
    supported_countries: Optional[List[str]] = None
    supported_payment_methods: Optional[List[str]] = None
    supported_currencies: Optional[List[str]] = None
    supported_card_brands: Optional[List[str]] = None
    supported_services: Optional[List[str]] = None
    payin_fee_percent: Optional[float] = None
    payout_fee_percent: Optional[float] = None
    payin_success_rate: Optional[float] = None
    payout_success_rate: Optional[float] = None

class PspCreate(PspBase):
    pass

class Psp(PspBase):
    id: uuid.UUID

class AiConfig(BaseModel):
    success_rate_weight: float
    cost_weight: float
    speed_weight: float

# --- Simplified Invitation Models ---
class InvitationCreate(BaseModel):
    email: EmailStr
    role: Literal['merchant', 'admin']

class MessageResponse(BaseModel):
    message: str
# --- End of Pydantic models ---

app = FastAPI(
    title="WiseRoute AI Routing Engine API",
    description="This API provides intelligent, AI-powered payment routing to optimize for success rate, cost, and speed.",
    version="1.0.0",
)

# --- CORS Middleware ---
origins = [
    "http://localhost:5173",
    "https://ai-routing-v2.vercel.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Authorization", "X-API-Key"],
)
# --- End of CORS ---

# --- Supabase and Gemini Config ---
supabase: AsyncClient = AsyncClient(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
genai.configure(api_key=config.GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')
# --- End of Config ---

# --- Authentication Dependencies ---
async def get_current_user(authorization: Annotated[str | None, Header()] = None):
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ")[1]
    try:
        user_response = await supabase.auth.get_user(jwt=token)
        return user_response.user
    except Exception:
        return None

async def get_user_from_api_key(x_api_key: Annotated[str | None, Header()] = None):
    if not x_api_key:
        return None
    profile_response = await supabase.from_("profiles").select("id").eq("api_key", x_api_key).single().execute()
    if not profile_response.data:
        return None
    user_id = profile_response.data['id']
    try:
        user_response = await supabase.auth.admin.get_user_by_id(user_id)
        return user_response.user
    except Exception:
        return None

async def get_user_from_token_or_key(
    user_from_jwt: Annotated[dict | None, Depends(get_current_user)] = None,
    user_from_api_key: Annotated[dict | None, Depends(get_user_from_api_key)] = None
):
    if user_from_jwt:
        return user_from_jwt
    if user_from_api_key:
        return user_from_api_key
    raise HTTPException(status_code=401, detail="Not authenticated: No valid token or API key provided")

async def get_current_admin_user(current_user: Annotated[dict, Depends(get_current_user)]):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user_id = current_user.id
    response = await supabase.from_("profiles").select("role").eq("id", user_id).single().execute()

    if response.data and response.data.get("role") == "admin":
        return current_user
    raise HTTPException(status_code=403, detail="Forbidden: User is not an admin")
# --- End of Authentication ---

# --- Helper Functions ---
def get_card_brand_from_bin(card_bin: str) -> Optional[str]:
    if not card_bin: return None
    if card_bin.startswith('4'): return 'visa'
    elif card_bin.startswith('5'): return 'mastercard'
    elif card_bin.startswith(('34', '37')): return 'amex'
    return 'unknown'

async def get_psp_score(psp: dict, transaction: Transaction, live_success_rate: Optional[float], ai_config: dict) -> Optional[dict]:
    strategy_instruction = f"""You are an AI expert in payment routing. Your goal is to score this PSP on a scale of 0-100 for the given transaction.
Your decision should be guided by these strategic weights, which define what is most important:
- Success Rate Weight: {ai_config.get('success_rate_weight', 0.5)}
- Cost Weight: {ai_config.get('cost_weight', 0.3)}
- Speed Weight: {ai_config.get('speed_weight', 0.2)}
Use these weights as your primary guide, but also use your own reasoning. Consider all the data provided: the PSP's general performance, its real-time success rate for this specific route, and the context of the transaction itself (amount, country, etc.).
In the 'reason' field, provide a concise, expert justification for your score. Explain how the data and the strategic weights led to your decision.
"""
    if transaction.transaction_type == 'payin':
        fee_percent = psp.get('payin_fee_percent', 0)
        success_rate = psp.get('payin_success_rate', 0)
    else: # payout
        fee_percent = psp.get('payout_fee_percent', 0)
        success_rate = psp.get('payout_success_rate', 0)
    transaction_details = f"* Amount: {transaction.amount} {transaction.currency}\n * Country: {transaction.country}"
    if transaction.payment_method:
        transaction_details += f"\n * Payment Method: {transaction.payment_method}"
    if transaction.card_bin:
        transaction_details += f"\n * Card BIN: {transaction.card_bin}"
    historical_insights = ""
    if live_success_rate is not None:
        historical_insights = f"* Live Success Rate (this route, last 7 days): {live_success_rate:.1%}"
    prompt = f"""You are a world-class Payment Routing Analyst.
**Your Guiding Strategy:** {strategy_instruction}
**Transaction Details:**
{transaction_details}
**PSP Performance Data:**
* Name: {psp.get('name')}
* Overall Success Rate for this transaction type: {success_rate * 100:.1f}%
* Fee for this transaction type: {fee_percent}%
* Speed Score (0 to 1): {psp.get('speed_score')}
* Risk Score (0 to 1, higher is worse): {psp.get('risk_score')}
**Live Historical Insights:**
{historical_insights if historical_insights else "No recent transaction history for this specific route."}
IMPORTANT: Respond ONLY with a valid JSON object of the following structure:
{{
"score": <your_final_score_here>,
"reason": "<your_expert_justification_here>"
}}"""
    try:
        ai_response = await gemini_model.generate_content_async(prompt)
        cleaned_response_text = ai_response.text.strip().replace("```json", "").replace("```", "")
        score_data = json.loads(cleaned_response_text)
        return {"psp_id": psp.get('id'), "psp_name": psp.get('name'), "score": score_data.get('score'), "reason": score_data.get('reason')}
    except Exception as e:
        print(f"Error scoring PSP {psp.get('name')}: {e}")
        return None
# --- End of Helper Functions ---

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "AI Payment Routing Engine is running."}

# --- Admin Endpoints ---
# main.py

@app.get("/admin/users", response_model=List[AdminUser])
async def get_all_users(admin_user: Annotated[dict, Depends(get_current_admin_user)]):
    response = await supabase.auth.admin.list_users()
    # CORRECTED: The response from list_users() has a .users attribute
    # containing the list. The previous error was in a different function.
    # After re-checking the Supabase V2 library, the response IS an object,
    # and we need to return its .users attribute.
    return response.users

@app.post("/admin/invite", response_model=MessageResponse)
async def invite_user(invitation: InvitationCreate, admin_user: Annotated[dict, Depends(get_current_admin_user)]):
    try:
        # Use Supabase's built-in invitation method. It handles the token and email sending for us.
        # We pass the desired role in the 'data' option.
        await supabase.auth.admin.invite_user_by_email(
            email=invitation.email,
            options={'data': {'role': invitation.role}}
        )
        return MessageResponse(message=f"Invitation successfully sent to {invitation.email}.")
    except Exception as e:
        # If the user already exists, Supabase will raise an error.
        # We catch it and return a user-friendly message.
        if 'User already exists' in str(e):
            raise HTTPException(status_code=400, detail="A user with this email already exists.")
        
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An unexpected server error occurred.")

@app.get("/admin/psps", response_model=List[Psp])
async def get_all_psps(admin_user: Annotated[dict, Depends(get_current_admin_user)]):
    response = await supabase.from_("psps").select("*").order("id").execute()
    return response.data

@app.post("/admin/psps", response_model=Psp)
async def create_psp(psp: PspCreate, admin_user: Annotated[dict, Depends(get_current_admin_user)]):
    response = await supabase.from_("psps").insert(psp.model_dump()).select("*").single().execute()
    if not response.data: raise HTTPException(status_code=500, detail="Failed to create PSP.")
    return response.data

@app.put("/admin/psps/{psp_id}", response_model=Psp)
async def update_psp(psp_id: uuid.UUID, psp: PspBase, admin_user: Annotated[dict, Depends(get_current_admin_user)]):
    response = await supabase.from_("psps").update(psp.model_dump(exclude_unset=True)).eq("id", str(psp_id)).select("*").single().execute()
    if not response.data: raise HTTPException(status_code=404, detail=f"PSP with id {psp_id} not found.")
    return response.data

@app.get("/admin/users/{user_id}/ai-config", response_model=AiConfig)
async def get_user_ai_config(user_id: uuid.UUID, admin_user: Annotated[dict, Depends(get_current_admin_user)]):
    response = await supabase.from_("profiles").select("success_rate_weight, cost_weight, speed_weight").eq("id", user_id).single().execute()
    if not response.data:
        raise HTTPException(status_code=404, detail=f"AI config for user {user_id} not found.")
    return response.data

@app.put("/admin/users/{user_id}/ai-config", response_model=AiConfig)
async def update_user_ai_config(user_id: uuid.UUID, config: AiConfig, admin_user: Annotated[dict, Depends(get_current_admin_user)]):
    await supabase.from_("profiles").update(config.model_dump()).eq("id", user_id).execute()
    response = await supabase.from_("profiles").select("*").eq("id", user_id).single().execute()
    if not response.data:
        raise HTTPException(status_code=500, detail="Failed to update AI config for user.")
    return response.data

# --- Merchant Endpoints ---
@app.get("/api-key", response_model=ApiKeyResponse)
async def get_api_key(current_user: Annotated[dict, Depends(get_current_user)]):
    user_id = current_user.id
    response = await supabase.from_("profiles").select("api_key").eq("id", user_id).single().execute()
    if response.data:
        return ApiKeyResponse(api_key=response.data.get("api_key"))
    return ApiKeyResponse(api_key=None)

@app.post("/api-key/generate", response_model=ApiKeyResponse)
async def generate_api_key(current_user: Annotated[dict, Depends(get_current_user)]):
    user_id = current_user.id
    new_key = f"sk_{secrets.token_urlsafe(24)}"
    await supabase.from_("profiles").update({"api_key": new_key}).eq("id", user_id).execute()
    response = await supabase.from_("profiles").select("api_key").eq("id", user_id).single().execute()
    if not response.data:
        raise HTTPException(status_code=500, detail="Could not update or find API key after generation.")
    return ApiKeyResponse(api_key=new_key)

@app.get("/merchant/ai-config", response_model=AiConfig)
async def get_my_ai_config(current_user: Annotated[dict, Depends(get_current_user)]):
    user_id = current_user.id
    response = await supabase.from_("profiles").select("success_rate_weight, cost_weight, speed_weight").eq("id", user_id).single().execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="AI config not found for current user.")
    return response.data

@app.put("/merchant/ai-config", response_model=AiConfig)
async def update_my_ai_config(config: AiConfig, current_user: Annotated[dict, Depends(get_current_user)]):
    user_id = current_user.id
    await supabase.from_("profiles").update(config.model_dump()).eq("id", user_id).execute()
    response = await supabase.from_("profiles").select("*").eq("id", user_id).single().execute()
    if not response.data:
        raise HTTPException(status_code=500, detail="Failed to update AI config.")
    return response.data

@app.post("/route-transaction", response_model=RoutingResponse)
async def route_transaction(
    transaction: Transaction,
    current_user: Annotated[dict, Depends(get_user_from_token_or_key)]
):
    user_id = current_user.id
    config_response = await supabase.from_("profiles").select("*").eq("id", user_id).single().execute()
    if not config_response.data:
        raise HTTPException(status_code=500, detail="AI configuration for user not found.")
    ai_config = config_response.data
    await supabase.from_("transactions").upsert({
        "id": str(transaction.transaction_id), "user_id": user_id, "amount": transaction.amount,
        "currency": transaction.currency, "geo": transaction.country, "payment_method": transaction.payment_method,
    }).execute()
    psps_response = await supabase.from_("psps").select("*").eq("is_active", True).execute()
    all_active_psps = psps_response.data
    if not all_active_psps:
        return RoutingResponse(ranked_psps=[])
    compatible_psps = []
    card_brand = get_card_brand_from_bin(transaction.card_bin)
    for psp in all_active_psps:
        country_ok = transaction.country in psp.get("supported_countries", [])
        currency_ok = transaction.currency in psp.get("supported_currencies", [])
        pm_ok = transaction.payment_method in psp.get("supported_payment_methods", [])
        service_ok = transaction.transaction_type in psp.get("supported_services", [])
        card_brand_ok = True
        if transaction.payment_method == 'credit_card' and card_brand:
            card_brand_ok = card_brand in psp.get("supported_card_brands", [])
        if country_ok and currency_ok and pm_ok and card_brand_ok and service_ok:
            compatible_psps.append(psp)
    if not compatible_psps:
        return RoutingResponse(ranked_psps=[])
    psp_ids = [psp['id'] for psp in compatible_psps]
    seven_days_ago = datetime.now() - timedelta(days=7)
    history_query = supabase.from_("transactions").select("routed_psp_id, status").in_("routed_psp_id", psp_ids).eq("geo", transaction.country).gte("created_at", seven_days_ago.isoformat())
    if transaction.payment_method:
        history_query = history_query.eq("payment_method", transaction.payment_method)
    history_response = await history_query.execute()
    historical_data = history_response.data
    live_psp_stats = {}
    for psp_id in psp_ids:
        psp_transactions = [t for t in historical_data if t.get('routed_psp_id') == psp_id]
        successes = len([t for t in psp_transactions if t.get('status') and t.get('status').lower() == 'completed'])
        failures = len([t for t in psp_transactions if t.get('status') and t.get('status').lower() == 'failed'])
        total_attempts = successes + failures
        if total_attempts > 0:
            live_psp_stats[psp_id] = successes / total_attempts
    tasks = [get_psp_score(psp, transaction, live_psp_stats.get(psp.get('id')), ai_config) for psp in compatible_psps]
    results = await asyncio.gather(*tasks)
    all_scores = [res for res in results if res is not None]
    if not all_scores:
        return RoutingResponse(ranked_psps=[])
    sorted_psps = sorted(all_scores, key=lambda psp: psp['score'], reverse=True)
    top_psps = sorted_psps[:3]
    ranked_response_list = [RankedPsp(rank=i + 1, **psp) for i, psp in enumerate(top_psps)]
    if ranked_response_list:
        top_ranked_psp = ranked_response_list[0]
        await supabase.from_("transactions").update({"routed_psp_id": top_ranked_psp.psp_id, "status": "routed (AI choice)"}).eq("id", str(transaction.transaction_id)).execute()
    return RoutingResponse(ranked_psps=ranked_response_list)

@app.post("/update-transaction-status")
async def update_transaction_status(update_data: TransactionStatusUpdate):
    try:
        response = await supabase.from_("transactions").update({"status": update_data.status}).eq("id", str(update_data.transaction_id)).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail=f"Transaction with ID {update_data.transaction_id} not found.")
        return update_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update transaction status: {e}")

@app.get("/dashboard-stats", response_model=DashboardStats)
async def get_dashboard_stats(current_user: Annotated[dict, Depends(get_current_user)]):
    user_id = current_user.id
    twenty_four_hours_ago = datetime.now() - timedelta(days=1)
    try:
        response = await supabase.from_("transactions").select("amount, status").eq("user_id", user_id).gte("created_at", twenty_four_hours_ago.isoformat()).execute()
        if not response.data:
            return DashboardStats(total_volume_24h=0, total_transactions_24h=0, success_rate_24h=0, avg_speed="N/A")
        transactions = response.data
        total_volume = sum(t['amount'] for t in transactions if t.get('amount'))
        total_transactions = len(transactions)
        completed_transactions = len([t for t in transactions if t.get('status') and t.get('status').lower() == 'completed'])
        success_rate = (completed_transactions / total_transactions * 100) if total_transactions > 0 else 0
        return DashboardStats(total_volume_24h=total_volume, total_transactions_24h=total_transactions, success_rate_24h=success_rate, avg_speed="1.2s")
    except Exception as e:
        print(f"Error fetching dashboard stats: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch dashboard stats.")

@app.get("/transactions", response_model=PaginatedTransactionsResponse)
async def get_transactions(
    current_user: Annotated[dict, Depends(get_current_user)],
    page: int = 1,
    page_size: int = 10
):
    user_id = current_user.id
    try:
        offset = (page - 1) * page_size
        response = await supabase.from_("transactions") \
            .select("id, created_at, amount, currency, geo, status", count='exact') \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .range(offset, offset + page_size - 1) \
            .execute()
        transactions_data = response.data or []
        total_count = response.count or 0
        return PaginatedTransactionsResponse(
            transactions=transactions_data,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
    except Exception as e:
        print(f"Error fetching transactions: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch transactions.")