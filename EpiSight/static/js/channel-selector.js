
// epilepsy_diagnosis_model_tester/frontend/js/channel-selector.js
document.addEventListener('DOMContentLoaded', function() {
    // 定义通道配置
    const channelConfig = {
        bipolar: [
            'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fp1-F3', 
            'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F8', 'F8-T4', 
            'T4-T6', 'T6-O2', 'Fp1-F7', 'F7-T3', 'T3-T5', 
            'T5-O1', 'Fz-Cz', 'Cz-Pz'
        ],
        monopolar: [
            "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", 
            "C3", "C4", "Cz", "T3", "T5", "T4", "T6", 
            "P3", "P4", "Pz", "O1", "O2"
        ]
    };

    // 获取DOM元素
    const bipolarContainer = document.getElementById('bipolar-channels');
    const monopolarContainer = document.getElementById('monopolar-channels');
    
    // 存储已选择的数字
    let selectedNumbers = new Set();
    
    // 初始化通道选择器
    function initChannelSelectors(channels, container) {
        container.innerHTML = '';
        selectedNumbers.clear();
        
        channels.forEach(channel => {
            const channelOption = document.createElement('div');
            channelOption.className = 'channel-option';
            
            const label = document.createElement('label');
            label.className = 'block text-sm font-medium mb-1 text-gray-300';
            label.textContent = channel;
            
            const select = document.createElement('select');
            select.className = 'channel-select';
            select.dataset.channel = channel;
            
            // 添加默认选项
            const defaultOption = document.createElement('option');
            defaultOption.value = '';
            defaultOption.textContent = 'Choose a number';
            defaultOption.disabled = true;
            defaultOption.selected = true;
            select.appendChild(defaultOption);
            
            // 添加数字选项并设置默认值
            for (let i = 1; i <= 20; i++) {
                const option = document.createElement('option');
                option.value = i;
                option.textContent = i;
                // 设置默认值为从1开始递增
                if (i === channels.indexOf(channel) + 1) {
                    option.selected = true;
                    selectedNumbers.add(i.toString());
                }
                select.appendChild(option);
            }
            
            // 监听选择变化
            select.addEventListener('change', function() {
                validateChannelSelection();
            });
            
            channelOption.appendChild(label);
            channelOption.appendChild(select);
            container.appendChild(channelOption);
        });
    }
    
    // 验证通道选择
    function validateChannelSelection() {
        // 重置状态
        selectedNumbers.clear();
        let isValid = true;
        
        // 获取当前活动的通道容器
        const activeContainer = document.querySelector('#bipolar-channels:not(.hidden)') || 
                               document.querySelector('#monopolar-channels:not(.hidden)');
        
        // 检查所有选择
        const selects = activeContainer.querySelectorAll('select');
        selects.forEach(select => {
            const value = select.value;
            
            if (!value) {
                // 未选择
                select.classList.add('channel-error');
                isValid = false;
            } else if (selectedNumbers.has(value)) {
                // 重复选择
                select.classList.add('channel-error');
                isValid = false;
            } else {
                // 有效选择
                select.classList.remove('channel-error');
                selectedNumbers.add(value);
            }
        });
        
        // 更新全局验证状态
        window.channelSelectionValid = isValid;
        return isValid;
    }
    
    // 监听双极蒙太奇选项变化
    document.addEventListener('bipolarOptionChanged', function(e) {
        const isBipolar = e.detail.isBipolar;
        
        if (isBipolar) {
            bipolarContainer.classList.remove('hidden');
            monopolarContainer.classList.add('hidden');
            initChannelSelectors(channelConfig.bipolar, bipolarContainer);
        } else {
            monopolarContainer.classList.add('hidden');
            bipolarContainer.classList.add('hidden');
            monopolarContainer.classList.remove('hidden');
            initChannelSelectors(channelConfig.monopolar, monopolarContainer);
        }
    });
    
    // 监听通道选择验证请求
    document.addEventListener('validateChannelSelection', function() {
        validateChannelSelection();
    });
    
    // 初始化默认显示双极蒙太奇
    initChannelSelectors(channelConfig.bipolar, bipolarContainer);
    bipolarContainer.classList.remove('hidden');
    monopolarContainer.classList.add('hidden');
});
