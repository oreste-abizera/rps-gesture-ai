// Tab switching
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName).classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
    
    // Load data for specific tabs
    if (tabName === 'visualize') {
        loadVisualizations();
    } else if (tabName === 'monitor') {
        loadMonitoring();
        startMonitoringInterval();
    }
}

// Image preview
function previewImage(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const preview = document.getElementById('imagePreview');
            preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
        };
        reader.readAsDataURL(file);
    }
}

// Predict image
async function predictImage() {
    const fileInput = document.getElementById('imageInput');
    const resultDiv = document.getElementById('predictionResult');
    
    if (!fileInput.files[0]) {
        resultDiv.innerHTML = '<div class="error">Please select an image first</div>';
        return;
    }
    
    const formData = new FormData();
    formData.append('image', fileInput.files[0]);
    
    resultDiv.innerHTML = '<div class="info">Predicting...</div>';
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.error) {
            resultDiv.innerHTML = `<div class="error">${result.error}</div>`;
            return;
        }
        
        const confidencePercent = (result.confidence * 100).toFixed(2);
        let probabilitiesHtml = '';
        for (const [label, prob] of Object.entries(result.class_probabilities)) {
            const probPercent = (prob * 100).toFixed(2);
            probabilitiesHtml += `
                <div class="probability-item">
                    <div class="label">${label.charAt(0).toUpperCase() + label.slice(1)}</div>
                    <div class="value">${probPercent}%</div>
                </div>
            `;
        }
        
        resultDiv.innerHTML = `
            <div class="prediction-result success">
                <h3>Predicted: ${result.predicted_class.charAt(0).toUpperCase() + result.predicted_class.slice(1)}</h3>
                <div class="confidence">Confidence: ${confidencePercent}%</div>
                <div class="probabilities">${probabilitiesHtml}</div>
            </div>
        `;
    } catch (error) {
        resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    }
}

// Upload files
document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('fileInput');
    const labelSelect = document.getElementById('labelSelect');
    const resultDiv = document.getElementById('uploadResult');
    
    if (!fileInput.files.length) {
        resultDiv.innerHTML = '<div class="error">Please select at least one file</div>';
        return;
    }
    
    if (!labelSelect.value) {
        resultDiv.innerHTML = '<div class="error">Please select a label</div>';
        return;
    }
    
    const formData = new FormData();
    formData.append('label', labelSelect.value);
    for (let i = 0; i < fileInput.files.length; i++) {
        formData.append('files', fileInput.files[i]);
    }
    
    resultDiv.innerHTML = '<div class="info">Uploading...</div>';
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.error) {
            resultDiv.innerHTML = `<div class="error">${result.error}</div>`;
        } else {
            resultDiv.innerHTML = `<div class="success">${result.message}</div>`;
            fileInput.value = '';
            labelSelect.value = '';
        }
    } catch (error) {
        resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    }
});

// Trigger retraining
async function triggerRetrain() {
    const statusDiv = document.getElementById('retrainStatus');
    const progressDiv = document.getElementById('retrainProgress');
    const retrainBtn = document.getElementById('retrainBtn');
    
    retrainBtn.disabled = true;
    statusDiv.innerHTML = '<div class="info">Starting retraining...</div>';
    progressDiv.style.display = 'block';
    
    try {
        const response = await fetch('/api/retrain', {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.error) {
            statusDiv.innerHTML = `<div class="error">${result.error}</div>`;
            progressDiv.style.display = 'none';
            retrainBtn.disabled = false;
            return;
        }
        
        statusDiv.innerHTML = '<div class="info">Retraining started. This may take several minutes...</div>';
        
        // Poll for status
        const interval = setInterval(async () => {
            const statusResponse = await fetch('/api/retrain/status');
            const statusResult = await statusResponse.json();
            
            if (!statusResult.training_in_progress) {
                clearInterval(interval);
                progressDiv.style.display = 'none';
                retrainBtn.disabled = false;
                statusDiv.innerHTML = '<div class="success">Retraining completed successfully!</div>';
            }
        }, 2000);
        
    } catch (error) {
        statusDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
        progressDiv.style.display = 'none';
        retrainBtn.disabled = false;
    }
}

// Load visualizations
async function loadVisualizations() {
    // Load stats
    const statsResponse = await fetch('/api/stats');
    const stats = await statsResponse.json();
    
    // Distribution chart
    const distributionData = [{
        x: ['Rock', 'Paper', 'Scissors'],
        y: [stats.raw_data.train.rock, stats.raw_data.train.paper, stats.raw_data.train.scissors],
        type: 'bar',
        marker: { color: ['#6ba3d8', '#7db3e8', '#8fc3f8'] }
    }];
    
    Plotly.newPlot('distributionChart', distributionData, {
        title: 'Training Data Distribution',
        xaxis: { title: 'Class' },
        yaxis: { title: 'Number of Images' }
    });
    
    // Metrics chart
    const metricsResponse = await fetch('/api/metrics');
    const metrics = await metricsResponse.json();
    
    if (metrics.length > 0) {
        const latest = metrics[0];
        const metricsData = [{
            x: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            y: [latest.accuracy, latest.precision, latest.recall, latest.f1_score],
            type: 'bar',
            marker: { color: '#4a90e2' }
        }];
        
        Plotly.newPlot('metricsChart', metricsData, {
            title: 'Latest Model Performance Metrics',
            yaxis: { range: [0, 1] }
        });
    }
    
    // Feature analysis charts
    const featureData1 = [{
        values: [stats.raw_data.train.rock, stats.raw_data.train.paper, stats.raw_data.train.scissors],
        labels: ['Rock', 'Paper', 'Scissors'],
        type: 'pie',
        marker: { colors: ['#6ba3d8', '#7db3e8', '#8fc3f8'] }
    }];
    
    Plotly.newPlot('featureChart2', featureData1, {
        title: 'Class Balance in Training Data'
    });
    
    // Image size distribution (simulated)
    const sizeData = [{
        x: ['Small', 'Medium', 'Large'],
        y: [stats.raw_data.train.total * 0.3, stats.raw_data.train.total * 0.5, stats.raw_data.train.total * 0.2],
        type: 'bar',
        marker: { color: '#4a90e2' }
    }];
    
    Plotly.newPlot('featureChart1', sizeData, {
        title: 'Image Size Distribution (Estimated)',
        xaxis: { title: 'Size Category' },
        yaxis: { title: 'Number of Images' }
    });
    
    // Training progress (if metrics available)
    if (metrics.length > 1) {
        const trainingData = [{
            x: metrics.map(m => m.model_version).reverse(),
            y: metrics.map(m => m.accuracy).reverse(),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Accuracy',
            line: { color: '#4a90e2' }
        }];
        
        Plotly.newPlot('featureChart3', trainingData, {
            title: 'Model Accuracy Over Training Iterations',
            xaxis: { title: 'Model Version' },
            yaxis: { title: 'Accuracy', range: [0, 1] }
        });
    }
}

// Load monitoring data
async function loadMonitoring() {
    try {
        // Health check
        const healthResponse = await fetch('/api/health');
        const health = await healthResponse.json();
        
        document.getElementById('modelStatus').textContent = health.model_loaded ? 'Online' : 'Offline';
        document.getElementById('modelStatus').style.color = health.model_loaded ? '#4caf50' : '#f44336';
        
        // Uptime data
        const uptimeResponse = await fetch('/api/uptime');
        const uptime = await uptimeResponse.json();
        
        document.getElementById('uptimePercentage').textContent = uptime.uptime_percentage.toFixed(2) + '%';
        document.getElementById('avgResponseTime').textContent = uptime.average_response_time.toFixed(2) + ' ms';
        document.getElementById('totalRequests').textContent = uptime.total_requests;
        
        // Response time chart
        if (uptime.recent_records.length > 0) {
            const responseTimeData = [{
                x: uptime.recent_records.map(r => r.timestamp),
                y: uptime.recent_records.map(r => r.response_time),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Response Time',
                line: { color: '#4a90e2' }
            }];
            
            Plotly.newPlot('responseTimeChart', responseTimeData, {
                title: 'Response Time Over Time',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Response Time (ms)' }
            });
            
            // Uptime chart
            const uptimeData = [{
                x: uptime.recent_records.map(r => r.timestamp),
                y: uptime.recent_records.map(r => r.status === 'up' ? 1 : 0),
                type: 'scatter',
                mode: 'lines',
                name: 'Uptime',
                line: { color: '#4caf50' },
                fill: 'tozeroy'
            }];
            
            Plotly.newPlot('uptimeChart', uptimeData, {
                title: 'Model Uptime History',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Status (1=Up, 0=Down)', range: [0, 1.2] }
            });
        }
    } catch (error) {
        console.error('Error loading monitoring data:', error);
    }
}

// Start monitoring interval
let monitoringInterval = null;
function startMonitoringInterval() {
    if (monitoringInterval) {
        clearInterval(monitoringInterval);
    }
    monitoringInterval = setInterval(loadMonitoring, 5001); // Update every 5 seconds
}

// Load initial data
window.addEventListener('load', function() {
    loadMonitoring();
});

