"""Pydantic schemas for the credit scoring API.

PredictionRequest exposes every raw application_train column the model was
trained on (excluding the TARGET label). Drift monitoring needs the full
input space — so we collect all 120 features + SK_ID_CURR rather than a
top-N subset.

Field ranges are business-driven (e.g. DAYS_BIRTH ∈ [-25550, -6570] = ages
18 to 70) rather than blindly fitted on training mins/maxes — this catches
nonsensical inputs at the API boundary.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Categorical literals — exhaustive list captured at training time.
# Mirrors models/app_train_categories.json. If the categorical vocabulary
# evolves, regenerate the JSON and update these literals.
# ---------------------------------------------------------------------------
ContractType = Literal["Cash loans", "Revolving loans"]
GenderCode = Literal["F", "M"]
YN = Literal["Y", "N"]
TypeSuite = Literal[
    "Children",
    "Family",
    "Group of people",
    "Other_A",
    "Other_B",
    "Spouse, partner",
    "Unaccompanied",
]
IncomeType = Literal[
    "Businessman",
    "Commercial associate",
    "Maternity leave",
    "Pensioner",
    "State servant",
    "Student",
    "Unemployed",
    "Working",
]
EducationType = Literal[
    "Academic degree",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
    "Secondary / secondary special",
]
FamilyStatus = Literal[
    "Civil marriage",
    "Married",
    "Separated",
    "Single / not married",
    "Unknown",
    "Widow",
]
HousingType = Literal[
    "Co-op apartment",
    "House / apartment",
    "Municipal apartment",
    "Office apartment",
    "Rented apartment",
    "With parents",
]
OccupationType = Literal[
    "Accountants",
    "Cleaning staff",
    "Cooking staff",
    "Core staff",
    "Drivers",
    "HR staff",
    "High skill tech staff",
    "IT staff",
    "Laborers",
    "Low-skill Laborers",
    "Managers",
    "Medicine staff",
    "Private service staff",
    "Realty agents",
    "Sales staff",
    "Secretaries",
    "Security staff",
    "Waiters/barmen staff",
]
WeekDay = Literal[
    "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"
]
OrganizationType = Literal[
    "Advertising",
    "Agriculture",
    "Bank",
    "Business Entity Type 1",
    "Business Entity Type 2",
    "Business Entity Type 3",
    "Cleaning",
    "Construction",
    "Culture",
    "Electricity",
    "Emergency",
    "Government",
    "Hotel",
    "Housing",
    "Industry: type 1",
    "Industry: type 10",
    "Industry: type 11",
    "Industry: type 12",
    "Industry: type 13",
    "Industry: type 2",
    "Industry: type 3",
    "Industry: type 4",
    "Industry: type 5",
    "Industry: type 6",
    "Industry: type 7",
    "Industry: type 8",
    "Industry: type 9",
    "Insurance",
    "Kindergarten",
    "Legal Services",
    "Medicine",
    "Military",
    "Mobile",
    "Other",
    "Police",
    "Postal",
    "Realtor",
    "Religion",
    "Restaurant",
    "School",
    "Security",
    "Security Ministries",
    "Self-employed",
    "Services",
    "Telecom",
    "Trade: type 1",
    "Trade: type 2",
    "Trade: type 3",
    "Trade: type 4",
    "Trade: type 5",
    "Trade: type 6",
    "Trade: type 7",
    "Transport: type 1",
    "Transport: type 2",
    "Transport: type 3",
    "Transport: type 4",
    "University",
    "XNA",
]
FondKapremontMode = Literal[
    "not specified", "org spec account", "reg oper account", "reg oper spec account"
]
HouseTypeMode = Literal["block of flats", "specific housing", "terraced house"]
WallsMaterialMode = Literal[
    "Block", "Mixed", "Monolithic", "Others", "Panel", "Stone, brick", "Wooden"
]
EmergencyStateMode = Literal["No", "Yes"]


# ---------------------------------------------------------------------------
# Request payload
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    """Raw application_train inputs for one client."""

    model_config = ConfigDict(extra="forbid")

    # Identity ---------------------------------------------------------------
    SK_ID_CURR: int = Field(ge=100_000, le=999_999_999, description="Client id")

    # Contract & demographics ------------------------------------------------
    NAME_CONTRACT_TYPE: ContractType
    CODE_GENDER: GenderCode
    FLAG_OWN_CAR: YN
    FLAG_OWN_REALTY: YN
    CNT_CHILDREN: int = Field(ge=0, le=20)
    AMT_INCOME_TOTAL: float = Field(gt=0)
    AMT_CREDIT: float = Field(gt=0)
    AMT_ANNUITY: Optional[float] = Field(default=None, gt=0)
    AMT_GOODS_PRICE: Optional[float] = Field(default=None, gt=0)
    NAME_TYPE_SUITE: Optional[TypeSuite] = None
    NAME_INCOME_TYPE: IncomeType
    NAME_EDUCATION_TYPE: EducationType
    NAME_FAMILY_STATUS: FamilyStatus
    NAME_HOUSING_TYPE: HousingType
    REGION_POPULATION_RELATIVE: float = Field(ge=0, le=1)
    DAYS_BIRTH: int = Field(le=-6570, ge=-25550, description="Negative days from now")
    DAYS_EMPLOYED: int = Field(
        ge=-25000,
        le=365_243,
        description="Negative days; 365243 sentinel = not employed",
    )
    DAYS_REGISTRATION: float = Field(le=0, ge=-25000)
    DAYS_ID_PUBLISH: int = Field(le=0, ge=-10000)
    OWN_CAR_AGE: Optional[float] = Field(default=None, ge=0, le=100)
    FLAG_MOBIL: int = Field(ge=0, le=1)
    FLAG_EMP_PHONE: int = Field(ge=0, le=1)
    FLAG_WORK_PHONE: int = Field(ge=0, le=1)
    FLAG_CONT_MOBILE: int = Field(ge=0, le=1)
    FLAG_PHONE: int = Field(ge=0, le=1)
    FLAG_EMAIL: int = Field(ge=0, le=1)
    OCCUPATION_TYPE: Optional[OccupationType] = None
    CNT_FAM_MEMBERS: float = Field(ge=1, le=20)
    REGION_RATING_CLIENT: int = Field(ge=1, le=3)
    REGION_RATING_CLIENT_W_CITY: int = Field(ge=1, le=3)
    WEEKDAY_APPR_PROCESS_START: WeekDay
    HOUR_APPR_PROCESS_START: int = Field(ge=0, le=23)
    REG_REGION_NOT_LIVE_REGION: int = Field(ge=0, le=1)
    REG_REGION_NOT_WORK_REGION: int = Field(ge=0, le=1)
    LIVE_REGION_NOT_WORK_REGION: int = Field(ge=0, le=1)
    REG_CITY_NOT_LIVE_CITY: int = Field(ge=0, le=1)
    REG_CITY_NOT_WORK_CITY: int = Field(ge=0, le=1)
    LIVE_CITY_NOT_WORK_CITY: int = Field(ge=0, le=1)
    ORGANIZATION_TYPE: OrganizationType

    # External scoring sources ----------------------------------------------
    EXT_SOURCE_1: Optional[float] = Field(default=None, ge=0, le=1)
    EXT_SOURCE_2: Optional[float] = Field(default=None, ge=0, le=1)
    EXT_SOURCE_3: Optional[float] = Field(default=None, ge=0, le=1)

    # Building characteristics (mostly nullable, ratios in [0, 1]) ----------
    APARTMENTS_AVG: Optional[float] = Field(default=None, ge=0, le=1)
    BASEMENTAREA_AVG: Optional[float] = Field(default=None, ge=0, le=1)
    YEARS_BEGINEXPLUATATION_AVG: Optional[float] = Field(default=None, ge=0, le=1)
    YEARS_BUILD_AVG: Optional[float] = Field(default=None, ge=0, le=1)
    COMMONAREA_AVG: Optional[float] = Field(default=None, ge=0, le=1)
    ELEVATORS_AVG: Optional[float] = Field(default=None, ge=0, le=1)
    ENTRANCES_AVG: Optional[float] = Field(default=None, ge=0, le=1)
    FLOORSMAX_AVG: Optional[float] = Field(default=None, ge=0, le=1)
    FLOORSMIN_AVG: Optional[float] = Field(default=None, ge=0, le=1)
    LANDAREA_AVG: Optional[float] = Field(default=None, ge=0, le=1)
    LIVINGAPARTMENTS_AVG: Optional[float] = Field(default=None, ge=0, le=1)
    LIVINGAREA_AVG: Optional[float] = Field(default=None, ge=0, le=1)
    NONLIVINGAPARTMENTS_AVG: Optional[float] = Field(default=None, ge=0, le=1)
    NONLIVINGAREA_AVG: Optional[float] = Field(default=None, ge=0, le=1)
    APARTMENTS_MODE: Optional[float] = Field(default=None, ge=0, le=1)
    BASEMENTAREA_MODE: Optional[float] = Field(default=None, ge=0, le=1)
    YEARS_BEGINEXPLUATATION_MODE: Optional[float] = Field(default=None, ge=0, le=1)
    YEARS_BUILD_MODE: Optional[float] = Field(default=None, ge=0, le=1)
    COMMONAREA_MODE: Optional[float] = Field(default=None, ge=0, le=1)
    ELEVATORS_MODE: Optional[float] = Field(default=None, ge=0, le=1)
    ENTRANCES_MODE: Optional[float] = Field(default=None, ge=0, le=1)
    FLOORSMAX_MODE: Optional[float] = Field(default=None, ge=0, le=1)
    FLOORSMIN_MODE: Optional[float] = Field(default=None, ge=0, le=1)
    LANDAREA_MODE: Optional[float] = Field(default=None, ge=0, le=1)
    LIVINGAPARTMENTS_MODE: Optional[float] = Field(default=None, ge=0, le=1)
    LIVINGAREA_MODE: Optional[float] = Field(default=None, ge=0, le=1)
    NONLIVINGAPARTMENTS_MODE: Optional[float] = Field(default=None, ge=0, le=1)
    NONLIVINGAREA_MODE: Optional[float] = Field(default=None, ge=0, le=1)
    APARTMENTS_MEDI: Optional[float] = Field(default=None, ge=0, le=1)
    BASEMENTAREA_MEDI: Optional[float] = Field(default=None, ge=0, le=1)
    YEARS_BEGINEXPLUATATION_MEDI: Optional[float] = Field(default=None, ge=0, le=1)
    YEARS_BUILD_MEDI: Optional[float] = Field(default=None, ge=0, le=1)
    COMMONAREA_MEDI: Optional[float] = Field(default=None, ge=0, le=1)
    ELEVATORS_MEDI: Optional[float] = Field(default=None, ge=0, le=1)
    ENTRANCES_MEDI: Optional[float] = Field(default=None, ge=0, le=1)
    FLOORSMAX_MEDI: Optional[float] = Field(default=None, ge=0, le=1)
    FLOORSMIN_MEDI: Optional[float] = Field(default=None, ge=0, le=1)
    LANDAREA_MEDI: Optional[float] = Field(default=None, ge=0, le=1)
    LIVINGAPARTMENTS_MEDI: Optional[float] = Field(default=None, ge=0, le=1)
    LIVINGAREA_MEDI: Optional[float] = Field(default=None, ge=0, le=1)
    NONLIVINGAPARTMENTS_MEDI: Optional[float] = Field(default=None, ge=0, le=1)
    NONLIVINGAREA_MEDI: Optional[float] = Field(default=None, ge=0, le=1)
    FONDKAPREMONT_MODE: Optional[FondKapremontMode] = None
    HOUSETYPE_MODE: Optional[HouseTypeMode] = None
    TOTALAREA_MODE: Optional[float] = Field(default=None, ge=0, le=1)
    WALLSMATERIAL_MODE: Optional[WallsMaterialMode] = None
    EMERGENCYSTATE_MODE: Optional[EmergencyStateMode] = None

    # Social circle ---------------------------------------------------------
    OBS_30_CNT_SOCIAL_CIRCLE: Optional[float] = Field(default=None, ge=0, le=500)
    DEF_30_CNT_SOCIAL_CIRCLE: Optional[float] = Field(default=None, ge=0, le=500)
    OBS_60_CNT_SOCIAL_CIRCLE: Optional[float] = Field(default=None, ge=0, le=500)
    DEF_60_CNT_SOCIAL_CIRCLE: Optional[float] = Field(default=None, ge=0, le=500)

    DAYS_LAST_PHONE_CHANGE: float = Field(le=0, ge=-15000)

    # Document flags --------------------------------------------------------
    FLAG_DOCUMENT_2: int = Field(ge=0, le=1)
    FLAG_DOCUMENT_3: int = Field(ge=0, le=1)
    FLAG_DOCUMENT_4: int = Field(ge=0, le=1)
    FLAG_DOCUMENT_5: int = Field(ge=0, le=1)
    FLAG_DOCUMENT_6: int = Field(ge=0, le=1)
    FLAG_DOCUMENT_7: int = Field(ge=0, le=1)
    FLAG_DOCUMENT_8: int = Field(ge=0, le=1)
    FLAG_DOCUMENT_9: int = Field(ge=0, le=1)
    FLAG_DOCUMENT_10: int = Field(ge=0, le=1)
    FLAG_DOCUMENT_11: int = Field(ge=0, le=1)
    FLAG_DOCUMENT_12: int = Field(ge=0, le=1)
    FLAG_DOCUMENT_13: int = Field(ge=0, le=1)
    FLAG_DOCUMENT_14: int = Field(ge=0, le=1)
    FLAG_DOCUMENT_15: int = Field(ge=0, le=1)
    FLAG_DOCUMENT_16: int = Field(ge=0, le=1)
    FLAG_DOCUMENT_17: int = Field(ge=0, le=1)
    FLAG_DOCUMENT_18: int = Field(ge=0, le=1)
    FLAG_DOCUMENT_19: int = Field(ge=0, le=1)
    FLAG_DOCUMENT_20: int = Field(ge=0, le=1)
    FLAG_DOCUMENT_21: int = Field(ge=0, le=1)

    # Credit bureau request volume -----------------------------------------
    AMT_REQ_CREDIT_BUREAU_HOUR: Optional[float] = Field(default=None, ge=0, le=500)
    AMT_REQ_CREDIT_BUREAU_DAY: Optional[float] = Field(default=None, ge=0, le=500)
    AMT_REQ_CREDIT_BUREAU_WEEK: Optional[float] = Field(default=None, ge=0, le=500)
    AMT_REQ_CREDIT_BUREAU_MON: Optional[float] = Field(default=None, ge=0, le=500)
    AMT_REQ_CREDIT_BUREAU_QRT: Optional[float] = Field(default=None, ge=0, le=500)
    AMT_REQ_CREDIT_BUREAU_YEAR: Optional[float] = Field(default=None, ge=0, le=500)


# ---------------------------------------------------------------------------
# Response payload
# ---------------------------------------------------------------------------
Decision = Literal["GRANTED", "REFUSED"]


class PredictionResponse(BaseModel):
    sk_id_curr: int
    probability_default: float = Field(ge=0, le=1)
    decision: Decision
    threshold: float = Field(ge=0, le=1)
    model_version: str
    client_known: bool = Field(
        description="True if SK_ID_CURR found in feature store; False otherwise."
    )


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    threshold: float
    n_features_expected: int
    metrics: dict
