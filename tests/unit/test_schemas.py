"""Schema validation tests — make sure the Pydantic ranges actually catch
the bad inputs the brief calls out (negative income, age out-of-range, etc.)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas import PredictionRequest


def test_valid_payload_accepted(valid_payload):
    PredictionRequest(**valid_payload)


def test_negative_income_rejected(valid_payload):
    valid_payload["AMT_INCOME_TOTAL"] = -100
    with pytest.raises(ValidationError):
        PredictionRequest(**valid_payload)


def test_zero_credit_rejected(valid_payload):
    valid_payload["AMT_CREDIT"] = 0
    with pytest.raises(ValidationError):
        PredictionRequest(**valid_payload)


def test_age_under_18_rejected(valid_payload):
    """DAYS_BIRTH ge -6570 enforces age ≤ 18 yr (closer to today)."""
    valid_payload["DAYS_BIRTH"] = -100  # roughly 3 months old
    with pytest.raises(ValidationError):
        PredictionRequest(**valid_payload)


def test_age_over_70_rejected(valid_payload):
    valid_payload["DAYS_BIRTH"] = -30000  # ~82 years
    with pytest.raises(ValidationError):
        PredictionRequest(**valid_payload)


def test_unknown_contract_type_rejected(valid_payload):
    valid_payload["NAME_CONTRACT_TYPE"] = "Some unknown type"
    with pytest.raises(ValidationError):
        PredictionRequest(**valid_payload)


def test_extra_field_rejected(valid_payload):
    valid_payload["__hacker_field__"] = 42
    with pytest.raises(ValidationError):
        PredictionRequest(**valid_payload)


def test_optional_fields_accept_null(valid_payload):
    valid_payload["EXT_SOURCE_1"] = None
    valid_payload["OWN_CAR_AGE"] = None
    PredictionRequest(**valid_payload)


def test_flag_must_be_zero_or_one(valid_payload):
    valid_payload["FLAG_DOCUMENT_3"] = 5
    with pytest.raises(ValidationError):
        PredictionRequest(**valid_payload)


def test_required_field_missing(valid_payload):
    del valid_payload["DAYS_BIRTH"]
    with pytest.raises(ValidationError):
        PredictionRequest(**valid_payload)
