let url = document.URL
if (url.endsWith('/index')) {
    // 移除 /index 并重定向
    url = url.slice(0, -6); // 移除最后的 6 个字符 (/index)
}
// epilepsy_diagnosis_model_tester/frontend/js/main.js
document.addEventListener('DOMContentLoaded', function() {
    // 初始化粒子特效
    initParticles();
    
    // Use Example按钮点击事件
    document.getElementById('use-example').addEventListener('click', function() {
        // 设置采样频率为256
        document.getElementById('sampling-rate').value = '256';
        
        // 选择Monopolar模式
        document.querySelector('input[name="bipolar"][value="false"]').checked = true;
        
        // 触发通道选择变化事件
        const event = new CustomEvent('bipolarOptionChanged', {
            detail: { isBipolar: false }
        });
        document.dispatchEvent(event);
        
        // 显示示例文件使用状态
        document.getElementById('file-upload-container').classList.add('hidden');
        document.getElementById('example-file-info').classList.remove('hidden');
        
        // 设置示例文件使用状态
        window.isUsingExample = true;
    });
    
    // 关闭示例按钮点击事件
    document.getElementById('close-example').addEventListener('click', function() {
        document.getElementById('file-upload-container').classList.remove('hidden');
        document.getElementById('example-file-info').classList.add('hidden');
        window.isUsingExample = false;
    });
    
    // 获取表单和相关元素
    const form = document.getElementById('model-test-form');
    const edfFileInput = document.getElementById('edf-file');
    const samplingRateInput = document.getElementById('sampling-rate');
    const bipolarRadios = document.querySelectorAll('input[name="bipolar"]');
    
    // 监听双极蒙太奇选择变化
    bipolarRadios.forEach(radio => {
        radio.addEventListener('change', function() {
            // 发送事件到channel-selector.js处理
            const event = new CustomEvent('bipolarOptionChanged', {
                detail: { isBipolar: this.value === 'true' }
            });
            document.dispatchEvent(event);
        });
    });
    
    // 文件上传处理
    edfFileInput.addEventListener('change', async function() {
        const file = this.files[0];
        if (!validateFile()) return;
        
        const submitBtn = form.querySelector('button[type="submit"]');
        submitBtn.disabled = true;
        showLoading('Uploading file...');
        
        try {
            const formData = new FormData();
            formData.append('file', file);
    
            const uploadResponse = await fetch(url+'/upload', {
                method: 'POST',
                body: formData
            });
    
            const { code, data: uid, msg } = await uploadResponse.json();
            
            if (code !== 200) {
                throw new Error(msg);
            }
            
            // 存储UID供后续分析使用
            form.dataset.uid = uid;
            submitBtn.disabled = false;
            hideLoading();
        } catch (error) {
            showPopupError(error.message);
            hideLoading();
        }
    });
    
    // 表单提交处理
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        clearErrors();
        
        if (!window.isUsingExample && !form.dataset.uid) {
            showError(edfFileInput, 'Please upload an EDF file first');
            return;
        }
        
        showLoading('Analyzing...');
        
        try {
            const analysisResponse = await fetch(url+'/process', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    uid: window.isUsingExample ? "example" : form.dataset.uid,
                    sample_rate: parseInt(samplingRateInput.value),
                    bipolar: document.querySelector('input[name="bipolar"]:checked').value === 'true',
                    channels: getSelectedChannels()
                })
            });
    
            const result = await analysisResponse.json();
            
            if (result.code === 200) {
                // 存储结果数据到localStorage并跳转
                localStorage.setItem('analysisResult', JSON.stringify(result.data));
                window.location.href = '/episight/report';
            } else {
                throw new Error(result.msg);
            }
        } catch (error) {
            showPopupError(error.message);
        } finally {
            hideLoading();
        }
    });
    
    // 文件验证
    function validateFile() {
        const file = edfFileInput.files[0];
        
        if (!file) {
            showError(edfFileInput, 'Please select a EDF file');
            return false;
        }
        
        const fileExt = file.name.split('.').pop().toLowerCase();
        if (fileExt !== 'edf') {
            showError(edfFileInput, 'Please select a valid EDF file');
            return false;
        }
        
        return true;
    }
    
    // 采样频率验证
    function validateSamplingRate() {
        const samplingRate = samplingRateInput.value;
        
        if (!samplingRate) {
            showError(samplingRateInput, 'Please enter a sampling rate');
            return false;
        }
        
        // 确保是整数
        if (!Number.isInteger(Number(samplingRate)) || Number(samplingRate) <= 0) {
            showError(samplingRateInput, 'Sampling rate must be an integer');
            return false;
        }
        
        return true;
    }
    
    // 通道选择验证
    function validateChannelSelection() {
        // 发送验证请求到channel-selector.js
        const event = new CustomEvent('validateChannelSelection');
        document.dispatchEvent(event);
        
        // 获取验证结果
        return window.channelSelectionValid;
    }
    
    // 显示错误信息
    function showError(element, message) {
        // 移除旧的错误信息
        const existingError = element.parentNode.querySelector('.error-message');
        if (existingError) {
            existingError.remove();
        }
        
        // 创建并添加新的错误信息
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        
        // 为元素添加错误样式
        element.classList.add('border-red-500');
        
        // 在元素后插入错误信息
        element.parentNode.appendChild(errorDiv);
        showPopupError(message);
    }
    
    // 清除所有错误信息
    function clearErrors() {
        // 移除所有错误消息
        document.querySelectorAll('.error-message').forEach(el => el.remove());
        
        // 移除所有错误样式
        document.querySelectorAll('.border-red-500').forEach(el => {
            el.classList.remove('border-red-500');
        });
    }
    function showPopupError(message) {
        // 创建错误提示弹窗
        const errorModal = document.createElement('div');
        errorModal.className = 'fixed inset-0 flex items-center justify-center z-50';
        errorModal.innerHTML = `
            <div class="fixed inset-0 bg-black opacity-50"></div>
            <div class="bg-gray-800 rounded-lg p-8 max-w-md mx-auto relative z-10 shadow-2xl border border-red-500">
                <div class="text-center">
                    <div class="text-red-400 text-5xl mb-4">
                        <i class="fas fa-exclamation-circle"></i>
                    </div>
                    <h3 class="text-2xl font-bold text-white mb-2">Error</h3>
                    <p class="text-gray-300 mb-6">${message}</p>
                    <button class="close-modal bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-6 rounded-lg transition">
                        OK
                    </button>
                </div>
            </div>
        `;
        
        // 添加关闭弹窗功能
        errorModal.querySelector('.close-modal').addEventListener('click', function() {
            errorModal.remove();
        });
        
        // 显示弹窗
        document.body.appendChild(errorModal);
    }
    
    // 显示加载状态
    function showLoading(content) {
        // 禁用提交按钮并显示加载状态
        const submitBtn = form.querySelector('button[type="submit"]');
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>' + content;
        submitBtn.classList.add('bg-blue-400');
        submitBtn.classList.remove('bg-blue-600', 'hover:bg-blue-700');
    }
    
    // 隐藏加载状态
    function hideLoading() {
        const submitBtn = form.querySelector('button[type="submit"]');
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fas fa-play-circle mr-2"></i>Start Analysis';
        submitBtn.classList.remove('bg-blue-400');
        submitBtn.classList.add('bg-blue-600', 'hover:bg-blue-700');
    }
    
    // 显示成功信息
    function showSuccess(result) {
        // 存储完整结果对象到本地存储
        localStorage.setItem('analysisResult', JSON.stringify(result));
        
        // 直接跳转到报告页面
        window.location.href = '/report';
    }
    
    
    // 创建成功提示弹窗
    const successModal = document.createElement('div');
    successModal.className = 'fixed inset-0 flex items-center justify-center z-50';
    successModal.innerHTML = `
        <div class="fixed inset-0 bg-black opacity-50"></div>
        <div class="bg-gray-800 rounded-lg p-8 max-w-md mx-auto relative z-10 shadow-2xl border border-blue-500">
            <div class="text-center">
                <div class="text-green-400 text-5xl mb-4">
                    <i class="fas fa-check-circle"></i>
                </div>
                <h3 class="text-2xl font-bold text-white mb-2">分析完成</h3>
                <p class="text-gray-300 mb-6">您的EEG数据已成功分析。完整功能正在开发中，敬请期待！</p>
                <button class="close-modal bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-6 rounded-lg transition">
                    确定
                </button>
            </div>
        </div>
    `;

    
    // 添加关闭弹窗功能
    successModal.querySelector('.close-modal').addEventListener('click', function() {
        successModal.remove();
    });
    
    // 初始化粒子特效
    function initParticles() {
        particlesJS('particles-js', {
            particles: {
                number: {
                    value: 80,
                    density: {
                        enable: true,
                        value_area: 800
                    }
                },
                color: {
                    value: '#3b82f6'
                },
                shape: {
                    type: 'circle',
                    stroke: {
                        width: 0,
                        color: '#000000'
                    }
                },
                opacity: {
                    value: 0.5,
                    random: false,
                    anim: {
                        enable: false
                    }
                },
                size: {
                    value: 3,
                    random: true,
                    anim: {
                        enable: false
                    }
                },
                line_linked: {
                    enable: true,
                    distance: 150,
                    color: '#60a5fa',
                    opacity: 0.4,
                    width: 1
                },
                move: {
                    enable: true,
                    speed: 2,
                    direction: 'none',
                    random: false,
                    straight: false,
                    out_mode: 'out',
                    bounce: false
                }
            },
            interactivity: {
                detect_on: 'canvas',
                events: {
                    onhover: {
                        enable: true,
                        mode: 'grab'
                    },
                    onclick: {
                        enable: true,
                        mode: 'push'
                    },
                    resize: true
                },
                modes: {
                    grab: {
                        distance: 140,
                        line_linked: {
                            opacity: 1
                        }
                    },
                    push: {
                        particles_nb: 4
                    }
                }
            },
            retina_detect: true
        });
    }
});

function getSelectedChannels() {
    const bipolar = document.querySelector('input[name="bipolar"]:checked').value === 'true';
    const channelContainer = bipolar ? document.getElementById('bipolar-channels') : document.getElementById('monopolar-channels');
    
    if (!channelContainer) {
        throw new Error('Channel container not found');
    }
    
    const selectedChannels = {};
    const inputs = Array.from(channelContainer.querySelectorAll('select'));
    const usedOrders = new Set();
    
    // 验证是否有重复或未填写的顺序
    let isValid = true;
    inputs.forEach(input => {
        if (!input.value || isNaN(input.value) || parseInt(input.value) <= 0) {
            isValid = false;
            input.classList.add('error');
        } else {
            const order = parseInt(input.value);
            if (usedOrders.has(order)) {
                isValid = false;
                input.classList.add('error');
            } else {
                usedOrders.add(order);
                input.classList.remove('error');
                const channelName = input.dataset.channel;
                if (channelName) {
                    selectedChannels[channelName] = order;
                }
            }
        }
    });
    
    if (!isValid) {
        throw new Error('Please fill in all channel orders and ensure they are unique.');
    }
    
    return selectedChannels;
}

// 全局变量，用于跨文件共享通道选择验证状态
window.channelSelectionValid = false;

