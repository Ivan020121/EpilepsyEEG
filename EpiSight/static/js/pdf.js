const data = JSON.parse(localStorage.getItem('analysisResult'));
function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours > 0 ? hours + 'h' : ''}${minutes > 0 ? minutes + 'm' : ''}${secs > 0 ? secs + 's' : ''}`;
}

function canvasToBase64(canvasId) {
    const canvas = document.getElementById(canvasId);
    return canvas ? canvas.toDataURL('image/png') : '';
}

async function getSVG(svgId) {
    try {
        // 1. 获取 SVG 元素
        const element = document.getElementById(svgId);
        if (!element) {
            throw new Error(`Element with ID ${svgId} not found.`);
        }

        // 2. 使用 html2canvas 截图元素
        const canvas = await html2canvas(element, {
            // 可选配置（根据需求调整）
            // 例如：设置窗口滚动位置为 0 以避免截取不全
            width: element.childNodes[0].getAttribute('width'),
            height: element.childNodes[0].getAttribute('height'),

            scale: window.devicePixelRatio,
            // 允许跨域资源（如果 SVG 包含外部资源）
            useCORS: true,
            // 允许 Canvas 被污染（如果 SVG 样式复杂）
            allowTaint: true,
        });

        // 3. 将 Canvas 转换为 Base64 PNG
        
        return canvas.toDataURL('image/png');
    } catch (error) {
        console.error('Error converting SVG:', error);
        throw error;
    }
}


async function get_dd() {
    return { 
        content: [
            {
                alignment: 'justify',
                columns: [
                    {
                        image: 'logo',
                        width: 60,
                    },
                    [
                        {
                            text: 'Placeholder',
                            style: 'nop',
                        },
                        {
                            width: '*',
                            text: 'Epilepsy Diagnosis Report',
                            style: 'header'
                        }
                    ]
                ]
            },
            {
                text: 'Diagnostic',
                style: 'title1'
            },
            {
                svg: '<svg viewBox="0 0 200 10"><line x1="0" y1="5" x2="200" y2="5" stroke="black" stroke-width="0.5"/></svg>',
                style: 'split',
                width: 514
                
            },
            {
                text: data.overview.seizure_duration > 0 ? 'Epilepsy like discharges were observed in the patien\'s EEG.' : 'No epileptic discharges were observed in the patient\'s EEG.',
                style: data.overview.seizure_duration > 0 ? 'bad_result' : 'good_result',
            },
            {
                text: 'Duration Summary',
                style: 'title2'
            },
            {
                table: {
                    widths: [ 100, 90, 90, 100, 90 ],
                    body: [
                        [{text: 'EEG\nRecording', style: 'table_head'}, {text: 'Normal', style: 'table_head'}, {text: 'Seizure', style: 'table_head'}, {text: 'Single Seizure\nAverage/Max/Min', style: 'table_head'}, {text: 'Seizur\nRatio', style: 'table_head'}],
                        [
                            {text: formatDuration(data.overview.normal_duration + data.overview.seizure_duration), alignment: 'center'},
                            {text: formatDuration(data.overview.normal_duration), alignment: 'center'},
                            {text: formatDuration(data.overview.seizure_duration), alignment: 'center'},
                            {text: formatDuration(data.overview.average_single_seizure_duration)+'/'+formatDuration(data.overview.max_single_seizure_duration)+'/'+formatDuration(data.overview.min_single_seizure_duration), alignment: 'center'},
                            {text: data.overview.seizure_ratio.toFixed(2)+'%', alignment: 'center'}]
                    ]
                },
            },
            {
                text: 'Raw EEG Signal',
                style: 'title2'
            },
            'The 1-second EEG segment illustrated below exhibits typical ictal epileptiform discharges',
            {
                image: canvasToBase64('eegSegmentChart'),
                width: 520,
                style: 'img'
            },
            
            {
                text: 'Time Domain Analysis',
                style: 'title1'
            },
            {
                svg: '<svg viewBox="0 0 200 10"><line x1="0" y1="5" x2="200" y2="5" stroke="black" stroke-width="0.5"/></svg>',
                style: 'split',
                width: 514
                
            },
            {
                text: 'Basic Time Domain Features',
                style: 'title2'
            },
            {
                table: {
                widths: [65, 65, 65, 65, 65, 65, 65],
                body: [
                        [{text: 'Channel', style: 'table_head'}, {text: 'Mean', style: 'table_head'}, {text: 'Std', style: 'table_head'}, {text: 'Max', style: 'table_head'}, {text: 'Min', style: 'table_head'}, {text: 'Median', style: 'table_head'}, {text: 'Range', style: 'table_head'}],
                        ...Object.entries(data.TimeDomainStats).map(([channel, values]) => [
                            {text: channel, alignment: 'center'},
                            {text: values[0].toFixed(3), alignment: 'center'},
                            {text: values[1].toFixed(3), alignment: 'center'},
                            {text: values[2].toFixed(3), alignment: 'center'},
                            {text: values[3].toFixed(3), alignment: 'center'},
                            {text: values[4].toFixed(3), alignment: 'center'},
                            {text: values[5].toFixed(3), alignment: 'center'}
                        ])
                    ]
                },
            },
            
            {
                text: 'Frequency Domain Analysis',
                style: 'title1'
            },
            {
                svg: '<svg viewBox="0 0 200 10"><line x1="0" y1="5" x2="200" y2="5" stroke="black" stroke-width="0.5"/></svg>',
                style: 'split',
                width: 514
                
            },
            {
                text: 'Power Spectral Density',
                style: 'title2'
            },
            {
                table: {
                widths: [65, 65, 65, 65, 65, 65, 65],
                body: [
                        [{text: 'Channel', style: 'table_head'}, {text: 'Total Power', style: 'table_head'}, {text: 'Delta', style: 'table_head'}, {text: 'Theta', style: 'table_head'}, {text: 'Alpha', style: 'table_head'}, {text: 'Beta', style: 'table_head'}, {text: 'Gamma', style: 'table_head'}],
                        ...Object.entries(data.FrequencyStats).map(([channel, values]) => [
                            {text: channel, alignment: 'center'},
                            {text: values[0].toFixed(3), alignment: 'center'},
                            {text: values[1].toFixed(3), alignment: 'center'},
                            {text: values[2].toFixed(3), alignment: 'center'},
                            {text: values[3].toFixed(3), alignment: 'center'},
                            {text: values[4].toFixed(3), alignment: 'center'},
                            {text: values[5].toFixed(3), alignment: 'center'}
                        ])
                    ]
                },
            },
            {
                text: 'Placeholder',
                color: 'white',
                margin: 8
            },
            {
                text: 'Power Spectral Density (PSD)',
                style: 'title2'
            },
            {
                image: canvasToBase64('psdChart'),
                width: 520,
                style: 'img'
            },
            {
                text: 'Frequency Band Power per Channel',
                style: 'title2'
            },
            {
                image: canvasToBase64('FBPChart'),
                width: 520,
                style: 'img'
            },
            
            {
                text: 'Placeholder',
                color: 'white',
                margin: 50
            },
            {
                text: 'Signal Intensity Analysis',
                style: 'title1'
            },
            {
                svg: '<svg viewBox="0 0 200 10"><line x1="0" y1="5" x2="200" y2="5" stroke="black" stroke-width="0.5"/></svg>',
                style: 'split',
                width: 514
                
            },
            {
                text: 'RMS Average',
                style: 'title2'
            },
            {
                table: {
                    widths: [ 250, 250 ],
                    body: [
                        [{text: 'Overall RMS Mean Across Channels (μV)', style: 'table_head'}, {text: 'Overall RMS Std Across Channels (μV)', style: 'table_head'}],
                        [{text: data.RMS[1].toFixed(3), alignment: 'center'},{text: data.RMS[2].toFixed(3), alignment: 'center'}]
                    ]
                },
            },
            {
                image: canvasToBase64('RMSAChart'),
                width: 520,
                style: 'img'
            },
            {
                text: 'RMS Average',
                style: 'title2'
            },
            {
                image: canvasToBase64('RMSChart'),
                width: 520,
                style: 'img'
            },
            
            {
                text: 'Placeholder',
                color: 'white',
                margin: 8
            },
            {
                text: 'Statistical Analysis',
                style: 'title1'
            },
            {
                svg: '<svg viewBox="0 0 200 10"><line x1="0" y1="5" x2="200" y2="5" stroke="black" stroke-width="0.5"/></svg>',
                style: 'split',
                width: 514
                
            },
            {
                text: 'Channel Covariance',
                style: 'title2'
            },
            {
                table: {
                    widths: [ 167, 167, 167 ],
                    body: [
                        [{text: 'Mean Covariance', style: 'table_head'}, {text: 'Max Covariance', style: 'table_head'}, {text: 'Min Covariance', style: 'table_head'}],
                        [{text: data.Covariance[0].toFixed(3), alignment: 'center'},{text: data.Covariance[1].toFixed(3), alignment: 'center'},{text: data.Covariance[2].toFixed(3), alignment: 'center'}]
                    ]
                },
            },
            {
                image: await getSVG('CovarianceChart'),
                width: 520,
                style: 'img'
            },
        ],
        styles: {
            nop: {
                fontSize: 7,
                color: 'white',
            },
            header: {
                fontSize: 20,
                bold: true,
                
            },
            bigger: {
                fontSize: 15,
                italics: true
            },
            title1: {
                fontSize: 18,
                bold: true,
                alignment: 'center',
                margin: [0,16,0,0]
            },
            title2: {
                fontSize: 15,
                alignment: 'left',
                margin: [0,8,0,12],
                bold: true,
            },
            bad_result: {
                color: '#dc2626',
                margin: [0,0,0,2],
                bold: true,
            },
            good_result: {
                color: '#0e6937',
                margin: [0,0,0,2],
                bold: true,
            },
            split: {
                alignment: 'center',
            },
            table_head: {
                bold: true,
                alignment: 'center'
            },
            img: {
                margin: [0,8,0,8],
            }
        },
        defaultStyle: {
            columnGap: 20
        },
        images: {
            logo: document.URL.slice(0,-7)+'/static/img/logo.png'           
        }
    }
}