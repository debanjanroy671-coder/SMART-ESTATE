from flask import Flask, render_template, send_from_directory

app = Flask(__name__, template_folder=".")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/style.css")
def style():
    return send_from_directory(".", "style.css")


if __name__ == "__main__":
    app.run(debug=True)
