// 初始化地图
const map = L.map('map-meteorology').setView([31.2825, 121.5064], 15); // 设置同济大学四平路校区为中心点，缩放级别为15


// 添加地图瓦片层
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

// 获取标注数据并动态添加到地图
fetch('/api/meteorology')
    .then(response => response.json())
    .then(data => {
        data.forEach(marker => {
            // 添加每个标注点到地图
            L.marker([marker.lat, marker.lng]).addTo(map)
                .bindPopup(marker.info); // 弹窗显示标注信息
        });
    });

// 添加图例
const legend = L.control({ position: 'bottomright' });
legend.onAdd = function () {
    const div = L.DomUtil.create('div', 'info legend');
    div.innerHTML = `
        <i style="background: #1f77b4; width: 12px; height: 12px; display: inline-block;"></i> 气象水文<br>
        <i style="background: #ff7f0e; width: 12px; height: 12px; display: inline-block;"></i> 积水径流<br>
        <i style="background: #2ca02c; width: 12px; height: 12px; display: inline-block;"></i> 雨污排水<br>
    `;
    return div;
};
legend.addTo(map);

