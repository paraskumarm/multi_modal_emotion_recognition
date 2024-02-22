document.addEventListener('DOMContentLoaded', function () {
    const webcamVideo = document.getElementById('webcamVideo');

    // Check if the browser supports getUserMedia
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        // Access the webcam
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function (stream) {
                // Display the webcam feed in the video element
                webcamVideo.srcObject = stream;
            })
            .catch(function (error) {
                console.error('Error accessing webcam:', error);
            });
    } else {
        console.error('getUserMedia is not supported in this browser');
    }
});
