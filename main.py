# main.py
import config
import google.generativeai as genai
import json
import asyncio
import secrets # New import for generating secure keys
from fastapi import FastAPI, HTTPException, Depends, Header # New imports
from fastapi.middleware.cors import CORSMiddleware
from supabase import AsyncClient
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Annotated # New import
import uuid
from datetime import datetime, timedelta

# --- Pydantic models (some are new) ---
class Transaction(BaseModel):
    transaction_id: uuid.UUID
    user_id: str
    amount: float
    currency: str
    country: str = Field(..., alias='geo')
    card_bin: Optional[str] = None
    payment_method: Optional[str] = None
    strategy: Literal["BALANCED", "MAX_SUCCESS", "MIN_COST"] = "BALANCED"
class TransactionStatusUpdate(BaseModel):
    transaction_id: uuid.UUID
    status: str
class RankedPsp(BaseModel):
    rank: int
    psp_id: str
    psp_name: str
    score: int
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
class ApiKeyResponse(BaseModel): # New model
    api_key: Optional[str] = None
# --- End of Pydantic models ---

app = FastAPI(title="AI Payment Routing Engine")

# --- CORS Middleware (Unchanged) ---
origins = [ "http://localhost:5173" ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- End of CORS ---

# --- Supabase and Gemini Config (Unchanged) ---
supabase: AsyncClient = AsyncClient(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
genai.configure(api_key=config.GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')
# --- End of Config ---

# NEW: Authentication dependency to protect routes
async def get_current_user(authorization: Annotated[str | None, Header()] = None):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")

    token = authorization.split(" ")[1]
    try:
        # The service key allows us to validate any user's token
        auth_client = AsyncClient(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
        user_response = await auth_client.auth.get_user(jwt=token)
        user = user_response.user
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user
    except Exception:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

async def get_psp_score(psp: dict, transaction: Transaction, live_success_rate: Optional[float]) -> Optional[dict]: # Unchanged
    # ... (function content is the same)
    strategy_instruction = "Your goal is to balance success rate, cost, and speed to provide the best overall value."
    if transaction.strategy == "MAX_SUCCESS":
        strategy_instruction = "Your primary goal is to maximize the chance of a successful transaction. Prioritize the highest success rate, even if fees are slightly higher."
    elif transaction.strategy == "MIN_COST":
        strategy_instruction = "Your primary goal is to minimize the cost of the transaction. Prioritize the lowest fee, even if the success rate is slightly lower."
    transaction_details = f"* Amount: {transaction.amount} {transaction.currency}\n    * Country: {transaction.country}"
    if transaction.payment_method:
        transaction_details += f"\n    * Payment Method: {transaction.payment_method}"
    if transaction.card_bin:
        transaction_details += f"\n    * Card BIN: {transaction.card_bin}"
    historical_insights = ""
    if live_success_rate is not None:
        historical_insights = f"* Live Success Rate (this route, last 7 days): {live_success_rate:.1%}"
    prompt = f"""You are a world-class Payment Routing Analyst...\n**Your Guiding Strategy:** {strategy_instruction}\n**Transaction Details:**\n    {transaction_details}\n**PSP Performance Data:**\n* Name: {psp.get('name')}\n* Overall Success Rate: {psp.get('success_rate') * 100:.1f}%\n* Fee: {psp.get('fee_percent')}%\n* Speed Score (0 to 1): {psp.get('speed_score')}\n* Risk Score (0 to 1, higher is worse): {psp.get('risk_score')}\n**Live Historical Insights:**\n{historical_insights if historical_insights else "No recent transaction history for this specific route."}\nIMPORTANT: Respond ONLY with a valid JSON object...{{\n"score": <your_score_here_0_to_100>,\n"reason": "<your_reason_here>"\n}}"""
    try:
        ai_response = await gemini_model.generate_content_async(prompt)
        cleaned_response_text = ai_response.text.strip().replace("```json", "").replace("```", "")
        score_data = json.loads(cleaned_response_text)
        return {"psp_id": psp.get('id'), "psp_name": psp.get('name'), "score": score_data.get('score'), "reason": score_data.get('reason')}
    except Exception as e:
        print(f"Error scoring PSP {psp.get('name')}: {e}")
        return None

# --- API Endpoints ---
@app.get("/")
def read_root(): # Unchanged
    return {"message": "AI Payment Routing Engine is running."}

# NEW: Add the secure endpoints for API keys
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
    # Generate a new secure key, e.g., "sk_xxxxxxxxxxxxxxxx"
    new_key = f"sk_{secrets.token_urlsafe(24)}"

    response, error = await supabase.from_("profiles") \
        .update({"api_key": new_key}) \
        .eq("id", user_id) \
        .execute()

    # Check for errors from the update operation
    if error and error.message:
         raise HTTPException(status_code=500, detail=f"Could not update API key: {error.message}")

    return ApiKeyResponse(api_key=new_key)

@app.post("/route-transaction", response_model=RoutingResponse)
async def route_transaction(transaction: Transaction): # Unchanged
    # ... (function content is the same)
    psps_response = await supabase.from_("psps").select("id, name, success_rate, fee_percent, speed_score, risk_score").eq("is_active", True).execute()
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
    tasks = [get_psp_score(psp, transaction, live_psp_stats.get(psp['id'])) for psp in active_psps]
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

@app.post("/update-transaction-status")
async def update_transaction_status(update_data: TransactionStatusUpdate): # Unchanged
    # ... (function content is the same)
    try:
        response = await supabase.from_("transactions").update({"status": update_data.status}).eq("id", str(update_data.transaction_id)).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail=f"Transaction with ID {update_data.transaction_id} not found.")
        return {"message": "Transaction status updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to update transaction status.")

@app.get("/dashboard-stats", response_model=DashboardStats)
async def get_dashboard_stats(): # Unchanged
    # ... (function content is the same)
    twenty_four_hours_ago = datetime.now() - timedelta(days=1)
    try:
        response = await supabase.from_("transactions").select("amount, status").gte("created_at", twenty_four_hours_ago.isoformat()).execute()
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
async def get_transactions(page: int = 1, page_size: int = 10): # Unchanged
    # ... (function content is the same)
    try:
        offset = (page - 1) * page_size
        response = await supabase.from_("transactions") \
            .select("id, created_at, amount, currency, geo, status", count='exact') \
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