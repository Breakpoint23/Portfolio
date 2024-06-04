document.addEventListener('DOMContentLoaded', () => {
  const textInput = document.getElementById('textInput');
  const log = document.getElementById('log');
  const clearButton = document.getElementById('clearButton');

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
        console.log("keyup",event.key);
    }
    updateLog();
  });

  clearButton.addEventListener('click', () => {
    keyLog = [];
    activeKeys = {};
    lastUpKeyData = null;
    updateLog();
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

