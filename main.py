# main.py
import config
import google.generativeai as genai
import json
import asyncio
import secrets
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from supabase import AsyncClient
from pydantic import BaseModel, Field
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
    success_rate: Optional[float] = None
    fee_percent: Optional[float] = None
    speed_score: Optional[float] = None
    risk_score: Optional[float] = None
    is_active: Optional[bool] = True

class PspCreate(PspBase):
    pass

class Psp(PspBase):
    id: uuid.UUID

class AiConfig(BaseModel):
    success_rate_weight: float
    cost_weight: float
    speed_weight: float
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
    allow_headers=["*", "Authorization", "X-API-Key"], # Add X-API-Key to allowed headers
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
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")
    token = authorization.split(" ")[1]
    try:
        user_response = await supabase.auth.get_user(jwt=token)
        user = user_response.user
        if user: return user
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    except Exception as e:
        print(f"Authentication error: {e}")
        raise HTTPException(status_code=401, detail="Could not validate credentials")

async def get_current_admin_user(current_user: Annotated[dict, Depends(get_current_user)]):
    user_id = current_user.id
    response = await supabase.from_("profiles").select("role").eq("id", user_id).single().execute()
    if response.data and response.data.get("role") == "admin":
        return current_user
    raise HTTPException(status_code=403, detail="Forbidden: User is not an admin")

# NEW: Dependency to get user from a permanent API key
async def get_user_from_api_key(x_api_key: Annotated[str | None, Header()] = None):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header missing")
    
    # Find the profile that matches this API key
    profile_response = await supabase.from_("profiles").select("id").eq("api_key", x_api_key).single().execute()
    if not profile_response.data:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    user_id = profile_response.data['id']
    
    # Get the full user object using the admin client
    try:
        user_response = await supabase.auth.admin.get_user_by_id(user_id)
        if user_response.user:
            return user_response.user
        raise HTTPException(status_code=404, detail="User not found for this API key")
    except Exception:
        raise HTTPException(status_code=500, detail="Could not retrieve user from API key")
# --- End of Authentication ---

# --- Helper Functions ---
async def get_psp_score(psp: dict, transaction: Transaction, live_success_rate: Optional[float], ai_config: dict) -> Optional[dict]:
    strategy_instruction = f"""Your goal is to score the PSP based on a weighted average of its performance metrics. The weights determine the importance of each factor. The current weights are:
- Success Rate: {ai_config.get('success_rate_weight', 0.5) * 100}%
- Cost (lower is better): {ai_config.get('cost_weight', 0.3) * 100}%
- Speed (higher is better): {ai_config.get('speed_weight', 0.2) * 100}%
Analyze the PSP's data against the transaction details and these weights to generate a score from 0 to 100."""
    
    transaction_details = f"* Amount: {transaction.amount} {transaction.currency}\n    * Country: {transaction.country}"
    if transaction.payment_method:
        transaction_details += f"\n    * Payment Method: {transaction.payment_method}"
    if transaction.card_bin:
        transaction_details += f"\n    * Card BIN: {transaction.card_bin}"
    
    historical_insights = ""
    if live_success_rate is not None:
        historical_insights = f"* Live Success Rate (this route, last 7 days): {live_success_rate:.1%}"
        
    prompt = f"""You are a world-class Payment Routing Analyst... (rest of prompt is the same)"""
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
# ... (All admin endpoints are the same)
@app.get("/admin/users", response_model=List[AdminUser])
async def get_all_users(admin_user: Annotated[dict, Depends(get_current_admin_user)]):
    # ...
    return []
@app.get("/admin/psps", response_model=List[Psp])
async def get_all_psps(admin_user: Annotated[dict, Depends(get_current_admin_user)]):
    # ...
    return []
@app.post("/admin/psps", response_model=Psp)
async def create_psp(psp: PspCreate, admin_user: Annotated[dict, Depends(get_current_admin_user)]):
    # ...
    return {}
@app.put("/admin/psps/{psp_id}", response_model=Psp)
async def update_psp(psp_id: uuid.UUID, psp: PspBase, admin_user: Annotated[dict, Depends(get_current_admin_user)]):
    # ...
    return {}
@app.get("/admin/ai-config", response_model=AiConfig)
async def get_ai_config(admin_user: Annotated[dict, Depends(get_current_admin_user)]):
    # ...
    return {}
@app.put("/admin/ai-config", response_model=AiConfig)
async def update_ai_config(config: AiConfig, admin_user: Annotated[dict, Depends(get_current_admin_user)]):
    # ...
    return {}

# --- Merchant Endpoints ---
@app.get("/api-key", response_model=ApiKeyResponse)
async def get_api_key(current_user: Annotated[dict, Depends(get_current_user)]):
    # ...
    return {}
@app.post("/api-key/generate", response_model=ApiKeyResponse)
async def generate_api_key(current_user: Annotated[dict, Depends(get_current_user)]):
    # ...
    return {}

# UPDATED: This endpoint now accepts either a JWT (for dashboard use) or an API Key (for server use)
@app.post("/route-transaction", response_model=RoutingResponse)
async def route_transaction(
    transaction: Transaction,
    # One of these two will be used, depending on what the client sends
    user_from_jwt: Annotated[dict | None, Depends(get_current_user)] = None,
    user_from_api_key: Annotated[dict | None, Depends(get_user_from_api_key)] = None
):
    current_user = user_from_jwt or user_from_api_key
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    user_id = current_user.id
    
    config_response = await supabase.from_("ai_config").select("*").eq("id", 1).single().execute()
    if not config_response.data:
        raise HTTPException(status_code=500, detail="AI configuration not found.")
    ai_config = config_response.data
    
    await supabase.from_("transactions").upsert({
        "id": str(transaction.transaction_id), "user_id": user_id, "amount": transaction.amount,
        "currency": transaction.currency, "geo": transaction.country, "payment_method": transaction.payment_method,
    }).execute()
    
    psps_response = await supabase.from_("psps").select("*").eq("is_active", True).execute()
    active_psps = psps_response.data
    if not active_psps:
        raise HTTPException(status_code=404, detail="No active PSPs found.")
    
    psp_ids = [psp['id'] for psp in active_psps]
    seven_days_ago = datetime.now() - timedelta(days=7)
    query = supabase.from_("transactions").select("routed_psp_id, status").in_("routed_psp_id", psp_ids).eq("geo", transaction.country).gte("created_at", seven_days_ago.isoformat())
    if transaction.payment_method:
        query = query.eq("payment_method", transaction.payment_method)
    history_response = await query.execute()
    historical_data = history_response.data
    live_psp_stats = {}
    for psp_id in psp_ids:
        psp_transactions = [t for t in historical_data if t.get('routed_psp_id') == psp_id]
        successes = len([t for t in psp_transactions if t.get('status') and t.get('status').lower() == 'completed'])
        failures = len([t for t in psp_transactions if t.get('status') and t.get('status').lower() == 'failed'])
        total_attempts = successes + failures
        if total_attempts > 0:
            live_psp_stats[psp_id] = successes / total_attempts
    
    tasks = [get_psp_score(psp, transaction, live_psp_stats.get(psp.get('id')), ai_config) for psp in active_psps]
    results = await asyncio.gather(*tasks)
    all_scores = [res for res in results if res is not None]
    if not all_scores:
        raise HTTPException(status_code=500, detail="Could not score any PSPs.")
    
    sorted_psps = sorted(all_scores, key=lambda psp: psp['score'], reverse=True)
    top_psps = sorted_psps[:3]
    ranked_response_list = [RankedPsp(rank=i + 1, **psp) for i, psp in enumerate(top_psps)]
    
    if ranked_response_list:
        top_ranked_psp = ranked_response_list[0]
        await supabase.from_("transactions").update({"routed_psp_id": top_ranked_psp.psp_id, "status": "routed (AI choice)"}).eq("id", str(transaction.transaction_id)).execute()
        
    return RoutingResponse(ranked_psps=ranked_response_list)

# ... (rest of endpoints are the same)
@app.post("/update-transaction-status")
async def update_transaction_status(update_data: TransactionStatusUpdate):
    return update_data
@app.get("/dashboard-stats", response_model=DashboardStats)
async def get_dashboard_stats(current_user: Annotated[dict, Depends(get_current_user)]):
    return DashboardStats(total_volume_24h=0, total_transactions_24h=0, success_rate_24h=0, avg_speed="N/A")
@app.get("/transactions", response_model=PaginatedTransactionsResponse)
async def get_transactions(current_user: Annotated[dict, Depends(get_current_user)], page: int = 1, page_size: int = 10):
    return PaginatedTransactionsResponse(transactions=[], total_count=0, page=1, page_size=10)
