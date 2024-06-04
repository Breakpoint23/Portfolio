document.addEventListener('DOMContentLoaded', () => {
  const textInput = document.getElementById('textInput');
  const log = document.getElementById('log');
  const clearButton = document.getElementById('clearButton');
  const submitButton = document.getElementById('submitButton');

  let keyLog = [];
  let activeKeys = {};
  let lastUpKeyData = null;

  textInput.addEventListener('keydown', (event) => {
    if (!activeKeys[event.key]) {
      activeKeys[event.key] = {
        key: event.key,
        keyCode: event.keyCode,
        pressTime: event.timeStamp,
        liftTime: null
      };
    }
  });

  textInput.addEventListener('keyup', (event) => {
    if (activeKeys[event.key]) {
      activeKeys[event.key].liftTime = event.timeStamp;
      keyLog.push(activeKeys[event.key]);
      lastUpKeyData = activeKeys[event.key];
      delete activeKeys[event.key];
    } else {
      keyLog.push({
        key: event.key,
        keyCode: event.keyCode,
        pressTime: lastUpKeyData ? lastUpKeyData.pressTime : null,
        liftTime: event.timeStamp
      });
      lastUpKeyData = null;
    }
    updateLog();
  });

  clearButton.addEventListener('click', () => {
    keyLog = [];
    activeKeys = {};
    lastUpKeyData = null;
    updateLog();
  });

  submitButton.addEventListener('click', () => {
    const jsonData = JSON.stringify(keyLog);
    // Assume 'url' is the API endpoint
    fetch('localhost:5000/data', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: jsonData
    })
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return response.json();
    })
    .then(data => {
      console.log('API response:', data);
    })
    .catch(error => {
      console.error('Error:', error);
    });
  });

  function updateLog() {
    log.innerHTML = '<h2>Key Press Log</h2>';
    keyLog.forEach(entry => {
      const logEntry = document.createElement('div');
      logEntry.textContent = `Key: ${entry.key}, KeyCode: ${entry.keyCode}, Press Time: ${entry.pressTime !== null ? entry.pressTime.toFixed(2) : 'N/A'}, Lift Time: ${entry.liftTime !== null ? entry.liftTime.toFixed(2) : 'N/A'}`;
      log.appendChild(logEntry);
    });
  }
});

