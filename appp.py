import pymysql
from flask import Flask, render_template, jsonify, request
import json
from static.model.MLq import FloodModel
from flask import Flask, render_template, request, jsonify
import base64
import io
import cv2
import numpy as np
from PIL import Image
from static.model.segformer import SegFormer_Segmentation

segformer = SegFormer_Segmentation()
app = Flask(__name__)

# 数据库初始化函数 (使用 pymysql)
def init_db():
    try:
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='123456',
            database='flood',  # 替换为你的数据库名称
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection
    except pymysql.MySQLError as e:
        raise ValueError(f"Database connection failed: {str(e)}")

# 首页
@app.route('/')
def home():
    return render_template('home.html')

# 态势感知
@app.route('/situation')
def situation():
    return render_template('situation.html')

# 研究区域
@app.route('/research')
def research():
    return render_template('research.html')

# API：返回动态标注数据
@app.route('/api/research')
def get_research():
    try:
        connection = init_db()
        with connection.cursor() as cursor:
            # 查询数据库中的研究区域数据
            query = "SELECT * FROM research"  # 表名为 'research'
            cursor.execute(query)
            markers = cursor.fetchall()
        
        # 关闭连接
        connection.close()
        
        return jsonify(markers)
    
    except Exception as e:
        return jsonify({'error': str(e)})

# 气象水文
@app.route('/meteorology')
def meteorology():
    return render_template('meteorology.html')

@app.route('/api/meteorology')
def get_meteorology():
    try:
        connection = init_db()
        with connection.cursor() as cursor:
            # 查询数据库中的研究区域数据
            query = "SELECT * FROM meteorology"  # 表名为 'research'
            cursor.execute(query)
            markers = cursor.fetchall()
        
        # 关闭连接
        connection.close()
        
        return jsonify(markers)
    
    except Exception as e:
        return jsonify({'error': str(e)})
# 积水径流
@app.route('/runoff')
def runoff():
    return render_template('runoff.html')

@app.route('/api/runoff')
def get_runoff():
    try:
        connection = init_db()
        with connection.cursor() as cursor:
            # 查询数据库中的研究区域数据
            query = "SELECT * FROM runoff"  # 表名为 'runoff'
            cursor.execute(query)
            markers = cursor.fetchall()
        
        # 关闭连接
        connection.close()
        
        return jsonify(markers)
    
    except Exception as e:
        return jsonify({'error': str(e)})

# 雨污排水
@app.route('/drainage')
def drainage():
    return render_template('drainage.html')

@app.route('/api/drainage')
def get_drainage():
    try:
        connection = init_db()
        with connection.cursor() as cursor:
            # 查询数据库中的研究区域数据
            query = "SELECT * FROM drainage"  # 表名为 'runoff'
            cursor.execute(query)
            markers = cursor.fetchall()
        
        # 关闭连接
        connection.close()
        
        return jsonify(markers)
    
    except Exception as e:
        return jsonify({'error': str(e)})

# 预报预警
@app.route('/forecast')
def forecast():
    return render_template('forecast.html')

@app.route('/analyze', methods=['GET'])
def analyze():
    table_name = request.args.get('flooddata')  # 表名

    if not table_name:
        return jsonify({"error": "Missing table name"}), 400

    try:
        # 使用 pymysql 初始化数据库连接
        connection = init_db()

        # 创建 FloodModel 实例并进行分析
        model = FloodModel(connection, table_name)
        model.clean_data()
        X_train, X_test, y_train, y_test, _, _ = model.prepare_data()
        rf_model = model.train_random_forest(X_train, y_train)
        analysis_result = model.evaluate_model(rf_model, X_train, X_test, y_train, y_test)

        # 生成图表
        cm_url, roc_url, feature_importance_url = model.generate_plots(
            analysis_result['cm'], analysis_result['fpr'],
            analysis_result['tpr'], analysis_result['roc_auc'],
            analysis_result['feature_importance']
        )

        connection.close()  # 关闭数据库连接

        return jsonify({
            "train_acc": analysis_result["train_acc"],
            "test_acc": analysis_result["test_acc"],
            "class_report": analysis_result["class_report"],
            "cm_url": cm_url,
            "roc_url": roc_url,
            "feature_importance_url": feature_importance_url
        })

    except Exception as e:
        return jsonify({"error": "Error occurred during analysis: " + str(e)}), 500
    
# 图片预测接口
@app.route('/predict-image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image = Image.open(image_file.stream)
        result_image = segformer.detect_image(image)

        # 将处理后的图片转为base64编码以便传输
        buffered = io.BytesIO()
        result_image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({'image': img_base64})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 视频处理接口
@app.route('/process-video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # 使用OpenCV处理视频
        video_stream = video_file.stream
        video_bytes = video_stream.read()
        np_array = np.frombuffer(video_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # 转换成RGB后处理
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame)
        result_frame = np.array(segformer.detect_image(frame_image))

        # 转回BGR格式并编码为视频流
        result_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', result_frame)
        result_video = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'video': result_video})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 应急联动
@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

# 用户中心
@app.route('/user')
def user():
    return render_template('user.html')

# 错误处理
@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

# 运行 Flask 应用
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
