from flask import Flask, render_template
from flask import jsonify
import json
from flask import Flask, render_template, jsonify
from static.model.ML import FloodModel
from static.scripts.hotspots_data import hotspots_data  


app = Flask(__name__)

# 首页
@app.route('/')
def home():
    return render_template('home.html')

#态势感知
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
        with open('api/research.json', 'r', encoding='utf-8') as file:
            markers = json.load(file)  # 从 JSON 文件加载数据
        return jsonify(markers)

    except Exception as e:
        return jsonify({'error': str(e)})

#气象水文
@app.route('/meteorology')
def meteorology():
    return render_template('meteorology.html')

@app.route('/api/meteorology')
def get_meteorology():
    try:
        with open('api/meteorology.json', 'r', encoding='utf-8') as file:
            markers = json.load(file)  # 从 JSON 文件加载数据
        return jsonify(markers)

    except Exception as e:
        return jsonify({'error': str(e)})

#积水径流  
@app.route('/runoff')
def runoff():
    return render_template('runoff.html')

@app.route('/api/runoff')
def get_runoff():
    try:
        with open('api/runoff.json', 'r', encoding='utf-8') as file:
            markers = json.load(file)  # 从 JSON 文件加载数据
        return jsonify(markers)

    except Exception as e:
        return jsonify({'error': str(e)})

#雨污排水
@app.route('/drainage')
def drainage():
    return render_template('drainage.html')

@app.route('/api/drainage')
def get_drainage():
    try:
        with open('api/drainage.json', 'r', encoding='utf-8') as file:
            markers = json.load(file)  # 从 JSON 文件加载数据
        return jsonify(markers)

    except Exception as e:
        return jsonify({'error': str(e)})
    
#预报预警
@app.route('/forecast')
def forecast():
    return render_template('forecast.html')

@app.route('/analyze', methods=['GET'])
def analyze():
    # 使用固定的数据路径
    data_path = r"E:\test\static\model\flood_data_generated.csv"
    
    # 创建FloodModel实例并进行分析
    model = FloodModel(data_path)
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

    return jsonify({
        "train_acc": analysis_result["train_acc"],
        "test_acc": analysis_result["test_acc"],
        "class_report": analysis_result["class_report"],
        "cm_url": cm_url,
        "roc_url": roc_url,
        "feature_importance_url": feature_importance_url
    })


#应急联动
@app.route('/analysis')
def analysis():
    # 传递热点数据到前端
    return render_template('analysis.html', hotspots_data=hotspots_data)




#用户中心
@app.route('/user')
def user():
    return render_template('user.html')

# 添加错误处理
@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

