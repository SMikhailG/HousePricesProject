def create_total_sf(df):
    """Создает признак общей площади дома TotalSF."""

    df = df.copy()

    df["TotalSF"] = (
        df["TotalBsmtSF"]
        + df["1stFlrSF"]
        + df["2ndFlrSF"]
    )

    return df


def create_house_age(df):
    """Создает признак возраста дома HouseAge."""

    df = df.copy()

    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    return df


def create_remod_age(df):
    """Создает признак времени после реконструкции RemodAge."""

    df = df.copy()

    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    return df


def create_total_bathrooms(df):
    """Создает признак общего количества ванных комнат TotalBathrooms."""

    df = df.copy()

    df["TotalBathrooms"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )

    return df


def build_features(df):
    """Создает все дополнительные признаки проекта."""

    df = create_total_sf(df)
    df = create_house_age(df)
    df = create_remod_age(df)
    df = create_total_bathrooms(df)

    return df