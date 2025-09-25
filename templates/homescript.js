
document.getElementById("sendBtn").addEventListener("click", sendMessage);
document.getElementById("userInput").addEventListener("keypress", function(e){
  if(e.key === "Enter") sendMessage();
});

function sendMessage() {
  const input = document.getElementById("userInput");
  const message = input.value.trim();
  if(!message) return;
  appendMessage("user", message);
  input.value = "";
  setTimeout(()=> {
    appendMessage("bot", "This is a placeholder bot response.");
  },500);
}

function appendMessage(sender, text){
  const chatBox = document.getElementById("chatBox");
  const msg = document.createElement("div");
  msg.className = `message ${sender}`;
  msg.textContent = text;
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}


const sidebar = document.getElementById("sidebar");
const container = document.querySelector(".container");
const sidebarToggle = document.getElementById("sidebarToggle");

sidebarToggle.addEventListener("click", () => {
  sidebar.classList.toggle("collapsed");
  container.classList.toggle("sidebar-collapsed");
});
