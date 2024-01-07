import pandas as pd
import pandas.io.formats.style
import tabulate


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
