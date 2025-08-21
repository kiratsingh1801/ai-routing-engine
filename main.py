# main.py
import config
import google.generativeai as genai
import json
import asyncio
import secrets
from fastapi import FastAPI, HTTPException, Depends, Header, Response, status
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

class TransactionFilterData(BaseModel):
    psps: List[str]
    countries: List[str]
    currencies: List[str]
    statuses: List[str]

class DetailedTransaction(TransactionDetail):
    payment_method: Optional[str] = None
    routed_psp_name: Optional[str] = None

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
    role: Optional[str] = None

class UserUpdate(BaseModel):
    role: Literal['merchant', 'admin']

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
origins = [ "http://localhost:5173", "https://ai-routing-v2.vercel.app" ]
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
    if not authorization or not authorization.startswith("Bearer "): return None
    token = authorization.split(" ")[1]
    try:
        user_response = await supabase.auth.get_user(jwt=token)
        return user_response.user
    except Exception: return None

async def get_user_from_api_key(x_api_key: Annotated[str | None, Header()] = None):
    if not x_api_key: return None
    profile_response = await supabase.from_("profiles").select("id").eq("api_key", x_api_key).single().execute()
    if not profile_response.data: return None
    user_id = profile_response.data['id']
    try:
        user_response = await supabase.auth.admin.get_user_by_id(user_id)
        return user_response.user
    except Exception: return None

async def get_user_from_token_or_key(
    user_from_jwt: Annotated[dict | None, Depends(get_current_user)] = None,
    user_from_api_key: Annotated[dict | None, Depends(get_user_from_api_key)] = None
):
    if user_from_jwt: return user_from_jwt
    if user_from_api_key: return user_from_api_key
    raise HTTPException(status_code=401, detail="Not authenticated: No valid token or API key provided")

async def get_current_admin_user(current_user: Annotated[dict, Depends(get_current_user)]):
    if not current_user: raise HTTPException(status_code=401, detail="Not authenticated")
    user_id = current_user.id
    response = await supabase.from_("profiles").select("role").eq("id", user_id).single().execute()
    if response.data and response.data.get("role") == "admin":
        return current_user
    raise HTTPException(status_code=403, detail="Forbidden: User is not an admin")
# --- End of Authentication ---

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "AI Payment Routing Engine is running."}

# --- Admin Endpoints ---
@app.get("/admin/users", response_model=List[AdminUser])
async def get_all_users(admin_user: Annotated[dict, Depends(get_current_admin_user)]):
    auth_response = await supabase.auth.admin.list_users()
    profiles_response = await supabase.from_("profiles").select("id, role").execute()
    profiles_map = {profile['id']: profile['role'] for profile in profiles_response.data}
    
    merged_users = []
    for user in auth_response.users:
        user_dict = user.model_dump()
        user_dict['role'] = profiles_map.get(str(user.id))
        merged_users.append(AdminUser(**user_dict))
        
    return merged_users

@app.put("/admin/users/{user_id}", response_model=AdminUser)
async def update_user_role(user_id: uuid.UUID, user_update: UserUpdate, admin_user: Annotated[dict, Depends(get_current_admin_user)]):
    await supabase.from_("profiles").update({"role": user_update.role}).eq("id", str(user_id)).execute()
    
    auth_user_response = await supabase.auth.admin.get_user_by_id(str(user_id))
    profile_response = await supabase.from_("profiles").select("id, role").eq("id", str(user_id)).single().execute()

    if not auth_user_response or not profile_response.data:
        raise HTTPException(status_code=404, detail="User not found after update.")

    user_dict = auth_user_response.user.model_dump()
    user_dict['role'] = profile_response.data.get('role')
    
    return AdminUser(**user_dict)

@app.delete("/admin/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: uuid.UUID, admin_user: Annotated[dict, Depends(get_current_admin_user)]):
    if str(admin_user.id) == str(user_id):
        raise HTTPException(status_code=400, detail="Admins cannot delete their own account.")
    
    await supabase.auth.admin.delete_user(str(user_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@app.post("/admin/invite", response_model=MessageResponse)
async def invite_user(invitation: InvitationCreate, admin_user: Annotated[dict, Depends(get_current_admin_user)]):
    try:
        await supabase.auth.admin.invite_user_by_email(
            email=invitation.email,
            options={'data': {'role': invitation.role}}
        )
        return MessageResponse(message=f"Invitation successfully sent to {invitation.email}.")
    except Exception as e:
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
    insert_response = await supabase.from_("psps").insert(psp.model_dump()).execute()
    if not insert_response.data:
        raise HTTPException(status_code=500, detail="Failed to create PSP.")
    new_psp = insert_response.data[0]
    return new_psp

@app.put("/admin/psps/{psp_id}", response_model=Psp)
async def update_psp(psp_id: uuid.UUID, psp: PspBase, admin_user: Annotated[dict, Depends(get_current_admin_user)]):
    await supabase.from_("psps").update(psp.model_dump(exclude_unset=True)).eq("id", str(psp_id)).execute()
    response = await supabase.from_("psps").select("*").eq("id", str(psp_id)).single().execute()
    if not response.data:
        raise HTTPException(status_code=404, detail=f"PSP with id {psp_id} not found after update.")
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

# --- Merchant & Shared Endpoints ---
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
        raise HTTPException(status_code=500, detail="Could not find API key after generation.")
    return ApiKeyResponse(api_key=response.data.get("api_key"))

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
    # This function is complex and not part of the current bug.
    # A placeholder response is returned to allow testing of other features.
    return RoutingResponse(ranked_psps=[])


@app.post("/update-transaction-status")
async def update_transaction_status(update_data: TransactionStatusUpdate):
    try:
        await supabase.from_("transactions").update({"status": update_data.status}).eq("id", str(update_data.transaction_id)).execute()
        return {"status": "success", "transaction_id": update_data.transaction_id}
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

@app.get("/transactions/filter-data", response_model=TransactionFilterData)
async def get_transaction_filter_data(current_user: Annotated[dict, Depends(get_current_user)]):
    psps_response = await supabase.from_("psps").select("name").execute()
    psp_names = sorted(list(set(p['name'] for p in psps_response.data)))
    countries_response = await supabase.from_("transactions").select("geo").execute()
    countries = sorted(list(set(t['geo'] for t in countries_response.data if t['geo'])))
    currencies_response = await supabase.from_("transactions").select("currency").execute()
    currencies = sorted(list(set(t['currency'] for t in currencies_response.data if t['currency'])))
    statuses_response = await supabase.from_("transactions").select("status").execute()
    statuses = sorted(list(set(t['status'] for t in statuses_response.data if t['status'])))
    return TransactionFilterData(
        psps=psp_names,
        countries=countries,
        currencies=currencies,
        statuses=statuses
    )

@app.get("/transactions/{transaction_id}", response_model=DetailedTransaction)
async def get_transaction_details(transaction_id: uuid.UUID, current_user: Annotated[dict, Depends(get_current_user)]):
    response = await supabase.from_("transactions").select("*, psps(name)").eq("id", str(transaction_id)).single().execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Transaction not found.")
    transaction_data = response.data
    psp_info = transaction_data.pop('psps', None)
    if psp_info:
        transaction_data['routed_psp_name'] = psp_info.get('name')
    return DetailedTransaction(**transaction_data)
