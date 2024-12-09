// 初始化地图
const map = new AMap.Map('map-research', {
    center: [121.50200, 31.283677], // 设置同济大学四平路校区为中心点
    zoom: 16.5 // 设置缩放级别
});

// 获取标注数据并动态添加到地图
fetch('/api/runoff')
    .then(response => response.json())
    .then(data => {
        data.forEach(marker => {
            // 创建标注
            const amapMarker = new AMap.Marker({
                position: new AMap.LngLat(marker.lng, marker.lat),
                title: marker.attribute || '未定义信息' // 悬浮提示
            });
            amapMarker.setMap(map);

            // 创建信息窗口
            const infoWindow = new AMap.InfoWindow({
                content: `<div style="padding: 5px;">${marker.attribute || '无信息'}</div>`, // 显示的内容
                offset: new AMap.Pixel(0, -30) // 调整偏移量
            });

            // 标注点击事件，打开信息窗口
            amapMarker.on('click', () => {
                infoWindow.open(map, amapMarker.getPosition());
            });

            // 如果需要默认显示信息窗口，可直接打开
            // infoWindow.open(map, amapMarker.getPosition());
        });
    });

// 添加图例
const legend = document.createElement('div');
legend.classList.add('info', 'legend');
legend.style.position = 'absolute';
legend.style.bottom = '20px';
legend.style.right = '20px';
legend.style.backgroundColor = 'white';
legend.style.padding = '10px';
legend.style.borderRadius = '5px';

legend.innerHTML = `
    <i style="background: #1f77b4; width: 12px; height: 12px; display: inline-block;"></i> 气象水文<br>
    <i style="background: #ff7f0e; width: 12px; height: 12px; display: inline-block;"></i> 积水径流<br>
    <i style="background: #2ca02c; width: 12px; height: 12px; display: inline-block;"></i> 雨污排水<br>
`;

document.getElementById('map-research').appendChild(legend);
