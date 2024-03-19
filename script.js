document.addEventListener("DOMContentLoaded", function() {
    // Get the form and file input element
    var form = document.getElementById('uploadForm');
    var fileInput = document.getElementById('file');

    // Add event listener for form submission
    form.addEventListener('submit', function(event) {
        // Check if a file is selected
        if (fileInput.files.length === 0) {
            alert('Please select a file.');
            event.preventDefault(); // Prevent form submission
        }
    });
});

