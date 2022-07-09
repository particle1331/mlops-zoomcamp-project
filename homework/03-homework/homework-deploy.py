from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from homework import main

DeploymentSpec(
    name="cron-schedule-deployment",
    flow_location="./homework.py",
    flow=main,
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="America/New_York"),
)
