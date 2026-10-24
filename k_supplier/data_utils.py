from pathlib import Path
from random import shuffle

from sklearn.preprocessing import MinMaxScaler
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd


def preprocess_data(X):
    X = X.copy()
    X = X.replace("?", np.nan)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.columns.difference(numeric_cols)
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())
    for col in categorical_cols:
        X[col] = X[col].fillna("unknown")
    X = pd.get_dummies(X, drop_first=True)
    X = X.apply(pd.to_numeric)
    X = MinMaxScaler().fit_transform(X)
    return X


def process_and_format_data(X,G,min_group_proportion=0):
    
    proportions = G.value_counts(normalize=True)
    valid_groups = proportions[proportions >= min_group_proportion].index

    keep = G.isin(valid_groups)
    X = X.loc[keep].reset_index(drop=True)
    G = G.loc[keep].reset_index(drop=True)

    # Ensure group labels are 0,1...,t-1
    G = pd.Series(pd.factorize(G, sort=True)[0])

    X = preprocess_data(X)

    return {
        "U":X,
        "G":G.values,
        "n":X.shape[0],
        "d":X.shape[1],
        "t":G.nunique()
    }


def load_synthetic(n, t, seed, min_group_size=10):

    n = int(n)
    t = int(t)
    d = 10

    group_seed, feature_seed = np.random.SeedSequence(seed).spawn(2)
    group_rng = np.random.default_rng(group_seed)
    feature_rng = np.random.default_rng(feature_seed)

    proportions = group_rng.dirichlet(np.ones(t))
    group_sizes = (
        group_rng.multinomial(n - (t*min_group_size), proportions)
        + min_group_size
    )
    G = np.repeat(np.arange(t, dtype=np.uint8), group_sizes)
    group_rng.shuffle(G)
    U = feature_rng.random((n, d), dtype=np.float32)

    return {
        "U": U,
        "G": G,
        "n": n,
        "d": d,
        "t": t,
        "group_sizes": group_sizes,
    }


def create_rvec(G,t,k,split,prng):

    groups, sizes = np.unique(G,return_counts=True)

    if split=="proportional":
        proportions = sizes / sizes.sum()
        _rvec = proportions * k
        _rvec = np.array([int(alpha) for alpha in _rvec])
        remaining = k - sum(_rvec)
        for i in range(remaining):
            _rvec[t-i-1]+=1
        for i in range(t):
            if _rvec[i]==0:
                _rvec[i]+=1
                _rvec[np.argmax(_rvec)]-=1

    elif split=="equitable":
        _rvec = np.array([k//t for i in range(t)])
        remaining = k - sum(_rvec)
        sampled = prng.choice(t,size=k%t,replace=False).tolist()
        for i in sampled:
            _rvec[i]+=1

    rvec = [0]*t
    for i in range(t):
        gi = groups[i]
        rvec[gi]=_rvec[i]

    return np.array(rvec)


def create_facilities(G,t,n_f,rvec,prng):

    if n_f == G.shape[0]:
        facilities = np.arange(G.shape[0], dtype=np.int32)
        GG_idx = [
            facilities[G == gi]
            for gi in range(t)
        ]
        GG_sizes = [len(group) for group in GG_idx]
        return facilities, GG_idx, GG_sizes

    facilities = []
    GG_idx = [[] for i in range(t)]

    # add minimum facilities for feasibility
    for gi in range(t):
        group_indices = np.flatnonzero(G == gi)
        alpha = rvec[gi]
        sample = prng.choice(group_indices,size=alpha,replace=False).tolist()
        facilities += sample
        GG_idx[gi] += sample

    # add remaining facilities 
    facilities = np.asarray(facilities)

    available_mask = np.ones(G.shape[0], dtype=bool)
    available_mask[facilities] = False
    available = np.flatnonzero(available_mask)

    remaining = n_f - len(facilities)
    additional = prng.choice(available,size=remaining,replace=False,shuffle=False)
    facilities = np.sort(
        np.concatenate((facilities, additional))
    )
    facility_groups = G[facilities]
    GG_idx = [
        facilities[facility_groups == gi]
        for gi in range(t)
    ]
    GG_sizes = [len(group) for group in GG_idx]
    return facilities, GG_idx, GG_sizes


def load_heart(min_group_proportion=0):
    X = fetch_ucirepo(id=45).data.features
    age = X.age.map(lambda x: "1" if x > X.age.median() else "0")
    sex = X.sex.astype(str)
    G = (age+sex).map(lambda x: int(x,2))
    X = X.drop(columns=['age','sex'])
    return process_and_format_data(X,G,min_group_proportion)


def load_student(min_group_proportion=0):
    X = fetch_ucirepo(id=320).data.features
    age = X.age.map(lambda x: "1" if x > X.age.median() else "0")
    sex = X.sex.map(lambda x: "1" if x == "M" else "0")
    G = (age + sex).map(lambda x: int(x,2))
    X = X.drop(columns=['age','sex'])
    return process_and_format_data(X,G,min_group_proportion)


def load_bank(min_group_proportion=0):
    X = fetch_ucirepo(id=222).data.features
    age = X.age.map(lambda x: "1" if x > X.age.median() else "0")
    marital = X.marital.map(lambda x: {
        "married":"00",
        "single":"01",
        "divorced":"10"
    }[x])
    G = (marital+age).map(lambda x: int(x,2))
    X = X.drop(columns=["age", "marital"])
    return process_and_format_data(X,G,min_group_proportion)


def load_census_adult(min_group_proportion=0):
    X = fetch_ucirepo(id=2).data.features
    age = X.age.map(lambda x: "1" if x > X.age.median() else "0")
    sex = X.sex.map(lambda x: "1" if x == "Male" else "0")
    race = X.race.map(lambda x: "1" if x == "White" else "0")
    nationality = X["native-country"].map(lambda x: "1" if x == "United-States" else "0")
    G = (age+sex+race+nationality).map(lambda x: int(x,2))
    X = X.drop(columns=["age","sex","race","native-country"])
    return process_and_format_data(X,G,min_group_proportion)


def load_credit_card(min_group_proportion=0):
    X = fetch_ucirepo(id=350).data.features
    age = X.X5.map(lambda x: "1" if x > X.X5.median() else "0")
    marital = X.X4.map(lambda x: "1" if x == 1 else "0")
    sex = (X.X2 - 1).astype(str)
    G = (age+sex+marital).map(lambda x: int(x,2))
    X = X.drop(columns=["X2","X4","X5"])
    return process_and_format_data(X,G,min_group_proportion)


def load_movielens_1m(min_group_proportion=0, root=None):
    if root is None:
        root = Path(__file__).resolve().parents[1] / "data" / "ml-1m"
    else:
        root = Path(root)

    users_path = root / "users.dat"
    ratings_path = root / "ratings.dat"
    movies_path = root / "movies.dat"

    users = pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1",
    )
    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        header=None,
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )

    rating_stats = ratings.groupby("user_id").agg(
        num_ratings=("rating", "size"),
        mean_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        min_rating=("rating", "min"),
        max_rating=("rating", "max"),
    ).fillna(0.0)

    genre_dummies = movies["genres"].str.get_dummies(sep="|")
    movie_genres = pd.concat([movies[["movie_id"]], genre_dummies], axis=1)
    ratings_with_genres = ratings.merge(movie_genres, on="movie_id", how="left")
    user_genre_profile = ratings_with_genres.groupby("user_id")[genre_dummies.columns.tolist()].mean().fillna(0.0)

    X = (
        users.merge(rating_stats, on="user_id", how="left")
        .merge(user_genre_profile, on="user_id", how="left")
        .fillna(0.0)
    )
    X["zip_prefix"] = X["zip_code"].astype(str).str[:3]

    age_bin = np.where(X["age"] <= 25, "0", "1")
    gender_bin = X["gender"].map({"F": "0", "M": "1"})
    G = (pd.Series(age_bin, index=X.index) + gender_bin).map(lambda s: int(s, 2))

    X = X.drop(columns=["user_id", "gender", "age", "zip_code"])
    return process_and_format_data(X, G, min_group_proportion)
