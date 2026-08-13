import numpy as np
import pandas as pd


def add_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create application-level engineered features."""
    df = df.copy()
    eps = 1e-6

    df["CREDIT_INCOME_RATIO"] = (
        df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + eps)
    )

    df["ANNUITY_INCOME_RATIO"] = (
        df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + eps)
    )

    df["CREDIT_ANNUITY_RATIO"] = (
        df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + eps)
    )

    df["GOODS_CREDIT_RATIO"] = (
        df["AMT_GOODS_PRICE"] / (df["AMT_CREDIT"] + eps)
    )

    df["INCOME_PER_PERSON"] = (
        df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + eps)
    )

    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

    df["EMPLOYED_YEARS"] = np.where(
        df["DAYS_EMPLOYED"] < 0,
        -df["DAYS_EMPLOYED"] / 365.25,
        np.nan,
    )

    df["EMPLOYED_AGE_RATIO"] = (
        df["EMPLOYED_YEARS"] / (df["AGE_YEARS"] + eps)
    )

    ext_cols = [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    ]

    df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
    df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)
    df["EXT_SOURCE_MAX"] = df[ext_cols].max(axis=1)
    df["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1)

    return df


def aggregate_bureau(
    bureau: pd.DataFrame,
    bureau_balance: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate bureau and bureau balance history by customer."""

    bb_agg = (
        bureau_balance.groupby("SK_ID_BUREAU")
        .agg(
            MONTHS_BALANCE_MIN=("MONTHS_BALANCE", "min"),
            MONTHS_BALANCE_MAX=("MONTHS_BALANCE", "max"),
            MONTHS_BALANCE_COUNT=("MONTHS_BALANCE", "count"),
        )
        .reset_index()
    )

    bureau_merged = bureau.merge(
        bb_agg,
        on="SK_ID_BUREAU",
        how="left",
    )

    bureau_agg = (
        bureau_merged.groupby("SK_ID_CURR")
        .agg(
            BURO_CREDIT_COUNT=("SK_ID_BUREAU", "count"),
            BURO_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
            BURO_DAYS_CREDIT_MIN=("DAYS_CREDIT", "min"),
            BURO_DAYS_CREDIT_MAX=("DAYS_CREDIT", "max"),
            BURO_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"),
            BURO_CREDIT_SUM_SUM=("AMT_CREDIT_SUM", "sum"),
            BURO_DEBT_MEAN=("AMT_CREDIT_SUM_DEBT", "mean"),
            BURO_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
            BURO_OVERDUE_MEAN=("AMT_CREDIT_SUM_OVERDUE", "mean"),
            BURO_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
            BURO_MONTHS_BALANCE_COUNT_MEAN=(
                "MONTHS_BALANCE_COUNT",
                "mean",
            ),
        )
        .reset_index()
    )

    bureau_status = (
        bureau.groupby(["SK_ID_CURR", "CREDIT_ACTIVE"])
        .size()
        .unstack(fill_value=0)
        .add_prefix("BURO_STATUS_")
        .add_suffix("_COUNT")
        .reset_index()
    )

    return bureau_agg.merge(
        bureau_status,
        on="SK_ID_CURR",
        how="left",
    )


def aggregate_previous(
    previous: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate previous loan applications by customer."""

    previous = previous.copy()

    previous["APP_CREDIT_RATIO"] = (
        previous["AMT_APPLICATION"]
        / previous["AMT_CREDIT"].replace(0, np.nan)
    )

    prev_agg = (
        previous.groupby("SK_ID_CURR")
        .agg(
            PREV_COUNT=("SK_ID_PREV", "count"),
            PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
            PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
            PREV_AMT_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
            PREV_APP_CREDIT_RATIO_MEAN=("APP_CREDIT_RATIO", "mean"),
            PREV_DAYS_DECISION_MEAN=("DAYS_DECISION", "mean"),
            PREV_CNT_PAYMENT_MEAN=("CNT_PAYMENT", "mean"),
        )
        .reset_index()
    )

    prev_status = (
        previous.groupby(["SK_ID_CURR", "NAME_CONTRACT_STATUS"])
        .size()
        .unstack(fill_value=0)
        .add_prefix("PREV_STATUS_")
        .add_suffix("_COUNT")
        .reset_index()
    )

    return prev_agg.merge(
        prev_status,
        on="SK_ID_CURR",
        how="left",
    )


def aggregate_installments(
    installments: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate installment payment history by customer."""

    installments = installments.copy()

    installments["PAYMENT_PERC"] = (
        installments["AMT_PAYMENT"]
        / installments["AMT_INSTALMENT"].replace(0, np.nan)
    )

    installments["PAYMENT_DIFF"] = (
        installments["AMT_INSTALMENT"]
        - installments["AMT_PAYMENT"]
    )

    installments["DPD"] = (
        installments["DAYS_ENTRY_PAYMENT"]
        - installments["DAYS_INSTALMENT"]
    ).clip(lower=0)

    installments["DBD"] = (
        installments["DAYS_INSTALMENT"]
        - installments["DAYS_ENTRY_PAYMENT"]
    ).clip(lower=0)

    return (
        installments.groupby("SK_ID_CURR")
        .agg(
            INSTAL_COUNT=("SK_ID_PREV", "count"),
            INSTAL_PAYMENT_PERC_MEAN=("PAYMENT_PERC", "mean"),
            INSTAL_PAYMENT_PERC_MIN=("PAYMENT_PERC", "min"),
            INSTAL_PAYMENT_DIFF_MEAN=("PAYMENT_DIFF", "mean"),
            INSTAL_PAYMENT_DIFF_SUM=("PAYMENT_DIFF", "sum"),
            INSTAL_DPD_MEAN=("DPD", "mean"),
            INSTAL_DPD_MAX=("DPD", "max"),
            INSTAL_DBD_MEAN=("DBD", "mean"),
            INSTAL_AMT_PAYMENT_MEAN=("AMT_PAYMENT", "mean"),
        )
        .reset_index()
    )


def build_v2_dataset(
    application: pd.DataFrame,
    bureau: pd.DataFrame,
    bureau_balance: pd.DataFrame,
    previous: pd.DataFrame,
    installments: pd.DataFrame,
) -> pd.DataFrame:
    """Build the V2 multi-table feature dataset."""

    bureau_agg = aggregate_bureau(
        bureau,
        bureau_balance,
    )

    previous_agg = aggregate_previous(
        previous,
    )

    installments_agg = aggregate_installments(
        installments,
    )

    df = application.copy()

    df = df.merge(
        bureau_agg,
        on="SK_ID_CURR",
        how="left",
    )

    df = df.merge(
        previous_agg,
        on="SK_ID_CURR",
        how="left",
    )

    df = df.merge(
        installments_agg,
        on="SK_ID_CURR",
        how="left",
    )

    return add_application_features(df)