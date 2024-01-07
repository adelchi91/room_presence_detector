import pandas as pd

def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def compute_global_figures(X, target):
    df = (
        pd.Series(
            {
                "contracts": int(X["contract_reference"].count()),
                "contracts with target": int(X[target].count()),
                "defaults": int(X[target].sum()),
                "default rate": X[target].mean(),
                "beg": X["application_date"].min().date(),
                "last with target": X.loc[~X[target].isna(), "application_date"].max().date(),
                "end": X["application_date"].max().date(),
            }
        )
        .to_frame()
        .T
    )
    return df.style.format(
        {
            "default rate": "{:.2%}",
        },
        thousands=",",
    )

def pprint_dataframe(X, **kwargs):
    if is_notebook():
        from IPython.display import display

        display(X)
    else:
        kwargs.setdefault("showindex", "never")
        kwargs.setdefault("headers", "keys")
        kwargs.setdefault("tablefmt", "psql")
        kwargs.setdefault("floatfmt", ".4f")
        kwargs.setdefault("intfmt", ",")
        if isinstance(X, pd.io.formats.style.Styler):
            X = X.data
        print(tabulate.tabulate(X, **kwargs))

# local functions
def print_cross_validation(cv):
    """
    prints cross-validation results in a dataframe, e.g. gini, brier, std...
    @param cv: cross-validation object
    @return: dataframe with relevant statistics
    """
    sort_key = f"mean_test_{cv.refit}" if isinstance(cv.refit, str) else "mean_test_score"

    df = (
        pd.json_normalize(cv.cv_results_["params"])
        .assign(
            **{
                f"{prefix}_{step}_{metric}": cv.cv_results_[f"{prefix}_{step}_{metric}"]
                for metric in cv.scoring
                for prefix in ("mean", "std")
                for step in ("test",)
            }
        )
        .sort_values(sort_key, ascending=False)
    )
    df["clf"] = df["clf"].apply(lambda clf: clf.version)
    try:
        df = df[~df["mean_test_auc"].isna()]
    except KeyError:
        df = df

    try:

        def apply_style(df):
            return (
                df.style.background_gradient(
                    axis=1,
                    vmin=0,
                    vmax=df["mean_test_roc_auc"].max(),
                    subset="mean_test_roc_auc",
                    # cmap=YC_PINKS,
                )
                .format(precision=4)
                .hide(axis="index")
            )

        pprint_dataframe(apply_style(df))
        pprint_dataframe(
            apply_style(
                df.groupby("clf", as_index=False).first().sort_values(f"mean_test_{selection_metric}", ascending=False)
            )
        )
    except KeyError:
        df = df
    return df
