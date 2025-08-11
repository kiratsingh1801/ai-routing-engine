# main.py

import config
import google.generativeai as genai
import json
import asyncio
from fastapi import FastAPI, HTTPException
from supabase import AsyncClient
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import uuid
from datetime import datetime, timedelta

# --- Pydantic Models (Our Data Contracts) ---
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


# --- Initialize our clients ---
supabase: AsyncClient = AsyncClient(
    config.SUPABASE_URL,
    config.SUPABASE_SERVICE_KEY
)
genai.configure(api_key=config.GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')


# --- Helper function to score one PSP asynchronously ---
async def get_psp_score(psp: dict, transaction: Transaction, live_success_rate: Optional[float]) -> Optional[dict]:
    strategy_instruction = "Your goal is to balance success rate, cost, and speed to provide the best overall value."
    if transaction.strategy == "MAX_SUCCESS":
        strategy_instruction = "Your primary goal is to maximize the chance of a successful transaction. Prioritize the highest success rate, even if fees are slightly higher."
    elif transaction.strategy == "MIN_COST":
        strategy_instruction = "Your primary goal is to minimize the cost of the transaction. Prioritize the lowest fee, even if the success rate is slightly lower."

    transaction_details = f"""
    * Amount: {transaction.amount} {transaction.currency}
    * Country: {transaction.country}"""
    if transaction.payment_method:
        transaction_details += f"\n    * Payment Method: {transaction.payment_method}"
    if transaction.card_bin:
        transaction_details += f"\n    * Card BIN: {transaction.card_bin}"

    historical_insights = ""
    if live_success_rate is not None:
        historical_insights = f"* Live Success Rate (this route, last 7 days): {live_success_rate:.1%}"

    prompt = f"""
    You are a world-class Payment Routing Analyst for the iGaming industry.
    
    **Your Guiding Strategy:** {strategy_instruction}

    Your task is to analyze a single transaction and a single Payment Service Provider (PSP) based on this strategy and return a recommendation score.

    **Transaction Details:**
    {transaction_details}

    **PSP Performance Data:**
    * Name: {psp.get('name')}
    * Overall Success Rate: {psp.get('success_rate') * 100:.1f}%
    * Fee: {psp.get('fee_percent')}%
    * Speed Score (0 to 1): {psp.get('speed_score')}
    * Risk Score (0 to 1, higher is worse): {psp.get('risk_score')}

    **Live Historical Insights:**
    {historical_insights if historical_insights else "No recent transaction history for this specific route."}

    IMPORTANT: Respond ONLY with a valid JSON object in the following format. Do not add any other text or explanations.

    {{
      "score": <your_score_here_0_to_100>,
      "reason": "<your_reason_here>"
    }}
    """
    try:
        ai_response = await gemini_model.generate_content_async(prompt)
        cleaned_response_text = ai_response.text.strip().replace("```json", "").replace("```", "")
        score_data = json.loads(cleaned_response_text)
        
        print(f"Scored {psp.get('name')} with strategy '{transaction.strategy}': {score_data.get('score')}")
        return {
            "psp_id": psp.get('id'),
            "psp_name": psp.get('name'),
            "score": score_data.get('score'),
            "reason": score_data.get('reason')
        }
    except Exception as e:
        print(f"Error scoring PSP {psp.get('name')}: {e}")
        return None


# --- Create the FastAPI App ---
app = FastAPI(title="AI Payment Routing Engine")


# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "AI Payment Routing Engine is running."}


@app.post("/route-transaction", response_model=RoutingResponse)
async def route_transaction(transaction: Transaction):
    print(f"Routing transaction {transaction.transaction_id} with strategy: {transaction.strategy}...")

    # 1. Fetch active PSPs
    psps_response = await supabase.from_("psps").select("id, name, success_rate, fee_percent, speed_score, risk_score").eq("is_active", True).execute()
    active_psps = psps_response.data
    if not active_psps:
        raise HTTPException(status_code=404, detail="No active PSPs found.")

    # 2. Fetch recent transaction history
    psp_ids = [psp['id'] for psp in active_psps]
    seven_days_ago = datetime.now() - timedelta(days=7)

    query = supabase.from_("transactions").select("routed_psp_id, status, geo, payment_method") \
        .in_("routed_psp_id", psp_ids) \
        .gte("created_at", seven_days_ago.isoformat())
    
    history_response = await query.execute()
    historical_data = history_response.data
    
    live_psp_stats = {}
    for psp_id in psp_ids:
        psp_transactions = [
            t for t in historical_data 
            if t.get('routed_psp_id') == psp_id and 
               t.get('geo') == transaction.country and
               t.get('payment_method') == transaction.payment_method
        ]
        
        # --- THIS IS THE FIX: Convert status to lowercase for comparison ---
        successes = len([t for t in psp_transactions if t.get('status') and t.get('status').lower() == 'completed'])
        failures = len([t for t in psp_transactions if t.get('status') and t.get('status').lower() == 'failed'])
        # --- END OF FIX ---

        total_attempts = successes + failures
        if total_attempts > 0:
            live_psp_stats[psp_id] = successes / total_attempts

    print(f"Live stats calculated for this route: {live_psp_stats}")

    # 3. Concurrently score all PSPs
    tasks = [get_psp_score(psp, transaction, live_psp_stats.get(psp['id'])) for psp in active_psps]
    results = await asyncio.gather(*tasks)
    all_scores = [res for res in results if res is not None]

    if not all_scores:
        raise HTTPException(status_code=500, detail="Could not score any PSPs.")

    # 4. Sort and rank the results
    sorted_psps = sorted(all_scores, key=lambda psp: psp['score'], reverse=True)
    top_psps = sorted_psps[:3]
    ranked_response_list = [RankedPsp(rank=i + 1, **psp) for i, psp in enumerate(top_psps)]

    # 5. Update the transaction record
    if ranked_response_list:
        top_ranked_psp = ranked_response_list[0]
        await supabase.from_("transactions").update({
            "routed_psp_id": top_ranked_psp.psp_id,
            "status": "routed (AI choice)"
        }).eq("id", str(transaction.transaction_id)).execute()
        print(f"Successfully updated transaction {transaction.transaction_id} with top choice: {top_ranked_psp.psp_name}")
    
    # 6. Return the final ranked list
    return RoutingResponse(ranked_psps=ranked_response_list)


@app.post("/update-transaction-status")
async def update_transaction_status(update_data: TransactionStatusUpdate):
    try:
        response = await supabase.from_("transactions").update({
            "status": update_data.status
        }).eq("id", str(update_data.transaction_id)).execute()

        if not response.data:
            raise HTTPException(
                status_code=404, 
                detail=f"Transaction with ID {update_data.transaction_id} not found."
            )

        print(f"Updated status for transaction {update_data.transaction_id} to '{update_data.status}'")
        return {"message": "Transaction status updated successfully."}

    except Exception as e:
        print(f"Error updating transaction status: {e}")
        raise HTTPException(status_code=500, detail="Failed to update transaction status.")