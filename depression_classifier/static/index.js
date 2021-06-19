function call_sentiment() {
    var sentiment = document.getElementById("sentence_sentiment").value;
    var xhttp = new XMLHttpRequest();

    xhttp.open("POST", "http://localhost:8000/pred");
    xhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    xhttp.send("sentiment=" + sentiment);
    xhttp.onreadystatechange = function() {
    if (this.readyState == 4 && this.status == 200) {
            var results = JSON.parse(this.responseText);
            document.getElementById("result").innerText = "";
            document.getElementById("result").innerText += 'Result : ' + results.classification + '\n Score : ' + results.score;
        }
    };
}

function login() {
    var email_list = new Map([
        ["rachel@gmail.com", '123'], 
        ["admin@test.com", '234'], 
        ["emergency@yahoo.com", '456']
    ])

    var email = document.getElementById("email").value;
    var password = document.getElementById("password").value;

    var gt_password = email_list.get(email);

    if (gt_password == undefined || gt_password != password) {
        alert("Wrong credentials");
    } else {
        window.location.href = 'http://localhost:8000/pred_form';
    }
}