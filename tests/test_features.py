import pandas as pd
import pytest

from src.features import (
    add_application_features,
    build_v2_dataset,
)


def test_add_application_features():
    df = pd.DataFrame(
        {
            "AMT_CREDIT": [500000.0],
            "AMT_INCOME_TOTAL": [100000.0],
            "AMT_ANNUITY": [25000.0],
            "AMT_GOODS_PRICE": [450000.0],
            "CNT_FAM_MEMBERS": [2.0],
            "DAYS_BIRTH": [-10957],
            "DAYS_EMPLOYED": [-1826],
            "EXT_SOURCE_1": [0.5],
            "EXT_SOURCE_2": [0.6],
            "EXT_SOURCE_3": [0.7],
        }
    )

    result = add_application_features(df)

    assert "CREDIT_INCOME_RATIO" in result.columns
    assert "AGE_YEARS" in result.columns
    assert "EXT_SOURCE_MEAN" in result.columns

    assert result.loc[0, "CREDIT_INCOME_RATIO"] == pytest.approx(
        5.0,
        rel=1e-5,
    )

    assert result.loc[0, "EXT_SOURCE_MEAN"] == pytest.approx(
        0.6
    )


def test_build_v2_dataset():
    application = pd.DataFrame(
        {
            "SK_ID_CURR": [100001],
            "TARGET": [0],
            "AMT_CREDIT": [500000.0],
            "AMT_INCOME_TOTAL": [100000.0],
            "AMT_ANNUITY": [25000.0],
            "AMT_GOODS_PRICE": [450000.0],
            "CNT_FAM_MEMBERS": [2.0],
            "DAYS_BIRTH": [-10957],
            "DAYS_EMPLOYED": [-1826],
            "EXT_SOURCE_1": [0.5],
            "EXT_SOURCE_2": [0.6],
            "EXT_SOURCE_3": [0.7],
        }
    )

    bureau = pd.DataFrame(
        {
            "SK_ID_CURR": [100001],
            "SK_ID_BUREAU": [200001],
            "DAYS_CREDIT": [-500],
            "AMT_CREDIT_SUM": [100000.0],
            "AMT_CREDIT_SUM_DEBT": [50000.0],
            "AMT_CREDIT_SUM_OVERDUE": [0.0],
            "CREDIT_DAY_OVERDUE": [0],
            "CREDIT_ACTIVE": ["Active"],
        }
    )

    bureau_balance = pd.DataFrame(
        {
            "SK_ID_BUREAU": [200001],
            "MONTHS_BALANCE": [-1],
        }
    )

    previous = pd.DataFrame(
        {
            "SK_ID_CURR": [100001],
            "SK_ID_PREV": [300001],
            "AMT_APPLICATION": [200000.0],
            "AMT_CREDIT": [180000.0],
            "AMT_ANNUITY": [10000.0],
            "DAYS_DECISION": [-200],
            "CNT_PAYMENT": [24.0],
            "NAME_CONTRACT_STATUS": ["Approved"],
        }
    )

    installments = pd.DataFrame(
        {
            "SK_ID_CURR": [100001],
            "SK_ID_PREV": [300001],
            "AMT_PAYMENT": [10000.0],
            "AMT_INSTALMENT": [10000.0],
            "DAYS_ENTRY_PAYMENT": [-30],
            "DAYS_INSTALMENT": [-30],
        }
    )

    result = build_v2_dataset(
        application,
        bureau,
        bureau_balance,
        previous,
        installments,
    )

    assert len(result) == 1

    assert result.loc[0, "BURO_CREDIT_COUNT"] == 1
    assert result.loc[0, "PREV_COUNT"] == 1
    assert result.loc[0, "INSTAL_COUNT"] == 1

    assert result.loc[0, "CREDIT_INCOME_RATIO"] == pytest.approx(
        5.0,
        rel=1e-5,
    )