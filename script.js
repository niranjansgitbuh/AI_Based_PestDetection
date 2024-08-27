document.getElementById('uploadForm').addEventListener('submit', async function(event) {
    event.preventDefault();
    
    const formData = new FormData();
    formData.append('image', document.getElementById('imageInput').files[0]);

    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    document.getElementById('result').innerHTML = `
        <h3>Prediction Result:</h3>
        <p>Pest/Disease: ${result.label}</p>
        <p>Treatment: ${result.treatment}</p>
    `;
});
