
### 深度学习服务部署的高并发解决方案

深度学习检测过程包含:加载模型、输入、检测、输出

问题:深度学习的检测方法为达到快速的目的需要用到GPU资源,而GPU资源是有限的,一台服务器同一时刻只能进行一次检测(多进程total_memory/detect_memory).当并发请求增多时,后到的请求只能等待前者检测完毕后才能进入检测.因此当请求进一步增多时,等待时间不断累计,最终会超过用户忍耐限度,影响用户体验.

解决思路:充分利用GPU的能力,一次检测,检测多次请求的图片,即批量检测.在此,引入redis内存数据库,一方面,web服务器接收到用户请求后,将请求数据(加上标记)放入redis队列中;另一方面,检测程序不断的从redis数据库中取一定量(batch_size)数据进行检测,并将返回结果也存储到redis数据库中;web服务器从redis数据库中取出对应的返回结果,发送至前端.

以下为flask_keras_redis的例子:

服务端(检测数据加入队列 从另一队列中取返回数据):

    # import the necessary packages
    from keras.preprocessing.image import img_to_array
    from keras.applications import imagenet_utils
    from PIL import Image
    import numpy as np
    import settings
    import helpers
    import flask
    import redis
    import uuid
    import time
    import json
    import io

    # initialize our Flask application and Redis server
    app = flask.Flask(__name__)
    
    #连接数据库
    ###########################################
    db = redis.StrictRedis(host=settings.REDIS_HOST,
        port=settings.REDIS_PORT, db=settings.REDIS_DB)
    ###########################################

    def prepare_image(image, target):
        # if the image mode is not RGB, convert it
        if image.mode != "RGB":
            image = image.convert("RGB")

        # resize the input image and preprocess it
        image = image.resize(target)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # return the processed image
        return image

    @app.route("/")
    def homepage():
        return "Welcome to the PyImageSearch Keras REST API!"

    @app.route("/predict", methods=["POST"])
    def predict():
        # initialize the data dictionary that will be returned from the
        # view
        data = {"success": False}

        # ensure an image was properly uploaded to our endpoint
        if flask.request.method == "POST":
            if flask.request.files.get("image"):
                # read the image in PIL format and prepare it for
                # classification
                image = flask.request.files["image"].read()
                image = Image.open(io.BytesIO(image))
                image = prepare_image(image,
                    (settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT))

                # ensure our NumPy array is C-contiguous as well,
                # otherwise we won't be able to serialize it
                image = image.copy(order="C")
                
                #生成唯一id,并将数据写入redis队列中
                ####################################################
                # generate an ID for the classification then add the
                # classification ID + image to the queue
                k = str(uuid.uuid4())
                image = helpers.base64_encode_image(image)
                d = {"id": k, "image": image}
                db.rpush(settings.IMAGE_QUEUE, json.dumps(d))
                ####################################################

                # keep looping until our model server returns the output
                # predictions
                while True:

                    #从队列中取出返回
                    ################################################
                    # attempt to grab the output predictions
                    output = db.get(k)
                    ################################################


                    # check to see if our model has classified the input
                    # image
                    if output is not None:
                        # add the output predictions to our data
                        # dictionary so we can return it to the client
                        output = output.decode("utf-8")
                        data["predictions"] = json.loads(output)

                        #从队列中删除已有返回的数据
                        ###############################################
                        # delete the result from the database and break
                        # from the polling loop
                        db.delete(k)
                        break
                        ################################################

                    # sleep for a small amount to give the model a chance
                    # to classify the input image
                    time.sleep(settings.CLIENT_SLEEP)

                # indicate that the request was a success
                data["success"] = True

        # return the data dictionary as a JSON response
        return flask.jsonify(data)

    # for debugging purposes, it's helpful to start the Flask testing
    # server (don't use this for production
    if __name__ == "__main__":
        print("* Starting web service...")
        app.run()



检测端(从队列中取出数据进行检测):

    # import the necessary packages
    from keras.applications import ResNet50
    from keras.applications import imagenet_utils
    import numpy as np
    import settings
    import helpers
    import redis
    import time
    import json
    from keras.models import load_model

    #连接数据库
    ##################################################
    # connect to Redis server
    db = redis.StrictRedis(host=settings.REDIS_HOST,
        port=settings.REDIS_PORT, db=settings.REDIS_DB)
    ###################################################

    def classify_process():
        # load the pre-trained Keras model (here we are using a model
        # pre-trained on ImageNet and provided by Keras, but you can
        # substitute in your own networks just as easily)
        print("* Loading model...")
        model = ResNet50(weights='imagenet')
        print("* Model loaded")

        # continually pool for new images to classify
        while True:

            #从队列中取批量图像数据,并合并为矩阵
            ###########################################################
            # attempt to grab a batch of images from the database, then
            # initialize the image IDs and batch of images themselves
            queue = db.lrange(settings.IMAGE_QUEUE, 0,
                settings.BATCH_SIZE - 1)
            imageIDs = []
            batch = None

            # loop over the queue
            for q in queue:
                # deserialize the object and obtain the input image
                q = json.loads(q.decode("utf-8"))
                image = helpers.base64_decode_image(q["image"],
                    settings.IMAGE_DTYPE,
                    (1, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH,
                        settings.IMAGE_CHANS))

                # check to see if the batch list is None
                if batch is None:
                    batch = image

                # otherwise, stack the data
                else:
                    batch = np.vstack([batch, image])

                # update the list of image IDs
                imageIDs.append(q["id"])
            #########################################################

            # check to see if we need to process the batch
            if len(imageIDs) > 0:
                # classify the batch
                print("* Batch size: {}".format(batch.shape))
                preds = model.predict(batch)
                results = imagenet_utils.decode_predictions(preds)

                # loop over the image IDs and their corresponding set of
                # results from our model
                for (imageID, resultSet) in zip(imageIDs, results):
                    # initialize the list of output predictions
                    output = []
                    
                    #将检测结果存放到list中存储
                    ###################################################
                    # loop over the results and add them to the list of
                    # output predictions
                    for (imagenetID, label, prob) in resultSet:
                        r = {"label": label, "probability": float(prob)}
                        output.append(r)
                    ####################################################

                    # store the output predictions in the database, using
                    # the image ID as the key so we can fetch the results
                    db.set(imageID, json.dumps(output))

                #将图片数据从队列中删除
                #####################################################
                # remove the set of images from our queue
                db.ltrim(settings.IMAGE_QUEUE, len(imageIDs), -1)
                #####################################################

            # sleep for a small amount
            time.sleep(settings.SERVER_SLEEP)

    # if this is the main thread of execution start the model server
    # process
    if __name__ == "__main__":
        classify_process()


其他思考:

1. 需要测试批量检测方法的极限,选择合适的batch_size;
2. 检测速度需要大于等于请求速度,因此需要限制在一段时间内的请求总数,防止数据库中的队列不断加长,最后爆掉;
3. 可以检测redis队列长度,超过一定的长度,进入睡眠时间进行缓冲,待队列中数据量降下来;
4. 可以在检测中采用多线程的方式进行检测,可以增加处理请求数;
5. 深度学习的检测过程最好不要加载在flask应用中,在生产环境中进行部署的时候,通常希望以多workers的方式启动,以解决web服务器对高并发的请求的处理能力,但由于GPU资源的限制,workers数只能很小,gunicorn/uwsgi的作用无法发挥出来,导致1+1<2的效果;

设想的架构应包含如下内容:

客户端 -> (SLB(请求分发) -> nginx(请求限制防止DOS攻击)) -> gunicorn/uwsgi(转发给application) -> flask < = > redis < = > detect server