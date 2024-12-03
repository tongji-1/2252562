function initialize() {
    // 创建地图实例
    var map = new AMap.Map('heatmap', {
        resizeEnable: true,
        center: [121.4737, 31.2304], // 设置地图中心坐标
        zoom: 15
    });

    // 创建热力图层
    var heatmap = new AMap.Heatmap(map);

    // 设置热力图样式
    heatmap.setOptions({
        radius: 20,
        opacity: [0, 0.8],
        gradient: {
            '0': 'green',
            '0.5': 'yellow',
            '1': 'red'
        }
    });

    // 模拟热点数据（你可以替换成从服务器获取的数据）
    var points = [
        {lng: 121.4737, lat: 31.2304, value: 100},
        {lng: 121.4740, lat: 31.2308, value: 80},
        {lng: 121.4725, lat: 31.2310, value: 120},
        {lng: 121.4730, lat: 31.2295, value: 60}
    ];

    // 转换为高德地图坐标格式
    var amapPoints = points.map(function (point) {
        return new AMap.LngLat(point.lng, point.lat);
    });

    // 添加数据到热力图
    heatmap.setDataSet({
        data: amapPoints,
        max: 200
    });
}

// 等待页面加载完成后初始化地图
window.onload = initialize;
