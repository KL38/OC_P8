"""Offline feature engineering module.

Contains heavy aggregation logic over bureau, previous_application,
POS_CASH_BALANCE, credit_card_balance, installments_payments. Used only by
scripts/build_feature_store.py — never imported by the runtime API.
"""
