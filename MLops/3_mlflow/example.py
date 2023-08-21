from mlflow import log_metric, log_param, log_artifact

if __name__ == '__main__':

    log_param("param", 1)

    log_metric("metric", 1)
    log_metric("metric", 2)
    log_metric("metric", 3)

    out_file_name = "output_file.txt"
    with open(out_file_name, "w") as f:
        f.write("content")
    log_artifact(out_file_name)
