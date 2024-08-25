from app import app
from app.granite import get_roadmaps
from flask import render_template, request


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/roadmap", methods=["POST"])
def get_roadmap():
    if not "msg" in request.form.keys() or request.form.get("msg") == "":
        err = {"code": 400, "title": "Bad Request! Empty field not supported."}
        return render_template("index.html", err=err)

    msg = request.form.get("msg")
    roadmap = get_roadmaps(msg)
    return render_template("roadmap.html", roadmaps=roadmap)
