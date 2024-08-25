from app import app

# from app.granite import
from flask import render_template, request


msg = ""
roadmap = [
    {
        "Machine Learning Fundamentals": "Dive into the fundamentals of machine learning, including supervised and unsupervised learning, and learn about popular algorithms such as linear regression, decision trees, and neural networks. Understand the concepts of training, validation, and testing, and learn how to evaluate model performance using appropriate metrics. Familiarize yourself with popular machine learning libraries and tools, such as Scikit-learn, TensorFlow, or PyTorch, which will enable you to build and evaluate machine learning models."
    },
    {
        "Hands-On Machine Learning Projects": "Gain practical experience in machine learning by working on personal projects that involve data collection, cleaning, and analysis. Apply machine learning techniques to real-world problems and build a portfolio that showcases your skills. Start with simple projects and gradually move on to more complex ones, ensuring that you build a strong foundation in machine learning before moving on to advanced topics. As you progress, challenge yourself by working on larger datasets and more sophisticated projects, pushing your skills to the next level."
    },
    {
        "Big Data and Predictive Modeling": "Explore big data technologies and predictive modeling techniques to expand your data science skill set. Learn about distributed computing frameworks, such as Hadoop or Spark, and their applications in data processing and analysis. Study advanced predictive modeling techniques, such as ensemble methods and deep learning, and their applications in real-world scenarios. Familiarize yourself with tools like Apache Spark, Hadoop, or cloud-based solutions like AWS SageMaker for working with big data and building predictive models."
    },
    {
        "Data Science Methodologies": "Develop a strong understanding of data science methodologies and best practices. Understand the importance of data cleaning, preprocessing, and exploratory data analysis (EDA), and learn how to design and implement robust data pipelines. Study data visualization techniques, such as heatmaps, scatter plots, and box plots, and learn how to communicate your findings effectively to stakeholders. Additionally, learn about ethical considerations in data science, including data privacy, security, and bias, and ensure that your work is conducted responsibly and ethically."
    },
    {
        "Data Science Communities and Resources": "Stay up-to-date with the latest trends and advancements in data science by following relevant blogs, attending workshops, and participating in online communities. Engage with data science professionals, ask questions, and share your experiences to learn from others and stay motivated. Utilize resources like online courses, books, and tutorials to deepen your understanding of data science concepts and techniques. By staying current and engaged with the data science community, you'll be better prepared to adapt to changes in the field and maintain a competitive edge in your career."
    },
]


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/roadmap", methods=["POST"])
def get_roadmap():
    if not "msg" in request.form.keys() or request.form.get("msg") == "":
        err = {"code": 400, "title": "Bad Request! Empty field not supported."}
        return render_template("index.html", err=err)

    msg = request.form.get("msg")
    print(msg)
    return render_template("roadmap.html", roadmaps=roadmap)
