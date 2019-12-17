import tensorflow as tf

class TestRun():

    def run(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # 텐서플로가 첫 번째 GPU만 사용하도록 제한
            print (gpus)
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        except RuntimeError as e:
        # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
            print(e)