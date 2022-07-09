import pandas as pd
import joblib

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import task, flow, get_run_logger

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, logger, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical, logger):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr, logger):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date):
    t = pd.to_datetime(date)
    train_path = f'./data/fhv_tripdata_{t.year}-0{t.month-2}.parquet'
    valid_path = f'./data/fhv_tripdata_{t.year}-0{t.month-1}.parquet'
    return train_path, valid_path


@flow
def main(date=None):
    train_path, val_path = get_paths(date).result()

    logger = get_run_logger()

    logger.info(f"Using train_path={train_path}, valid_path={val_path}")

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical, logger)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, logger, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical, logger).result()
    joblib.dump(lr, f'model-{date}.bin')    
    joblib.dump(dv, f'dv-{date}.b')    
    run_model(df_val_processed, categorical, dv, lr, logger)


from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    name="cron-schedule-deployment",
    flow=main,
    schedule=CronSchedule(cron="0 9 15 * *"),
    flow_runner=SubprocessFlowRunner(),
)


if __name__ == "__main__":
    main(date="2021-08-15")
