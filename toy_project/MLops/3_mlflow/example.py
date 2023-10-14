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


"""
log_param으로 실험에 대한 parameter를 기록해두고, log_metric으로는 accuracy 혹은 loss 같은 metric을 기록하여 차트로도 볼 수 있게 제공해주고, log_artifact는 학습된 모델과 같은 파일을 보관하게 됩니다.

 

mlflow를 로컬에서 실행했기 때문에 모두 파일로 관리되고 mlruns 디렉토리 밑에 보관된다고 위에서 설명했었는데, mlflow를 서버로 실행하는 경우에는 파일이 아니라 DB를 활용할 수 있게 제공되며 artifact는 클라우드 스토리지도 활용할 수 있게 제공되고 있습니다.

좀 더 구체적으로 DB의 경우에는 내부적으로 sqlalchemy를 활용하고 있기 때문에, sqlalchemy로 연동할 수 있는 DB engine은 모두 활용할 수 있고, artifact는 boto library를 활용하여 s3와 연동할 수 있습니다.

 

실험에 대해서 기록을 하기에는 매우 편한 ui를 제공하고 있고 그 외에도 서빙과 같은 기능도 제공한다고 하니 mlflow 한번쯤 써보시면 좋을 것 같습니다.

"""