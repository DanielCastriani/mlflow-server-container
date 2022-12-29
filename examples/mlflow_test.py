import mlflow
from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == "__main__":
    mlflow.set_tracking_uri(f"http://127.0.0.1:{os.environ.get('PORT')}")

    mlflow.start_run()

    mlflow.log_metric("metric", 321)

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    with open("outputs/test_file.txt", "w") as f:
        f.write("a,b,c\n")
        f.write("1,2,3\n")
        f.write("4,5,6\n")

    mlflow.log_artifact("outputs/test_file.txt", 'test')

    mlflow.end_run()

