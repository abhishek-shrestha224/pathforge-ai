from langchain_ibm import ChatWatsonx
from dotenv import load_dotenv
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain.schema.output_parser import StrOutputParser

# Example for fewshot prompting

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

# SkillGap Analysis

skillgap_examples = [
    {
        "input": """
    Current Skillset:
    I know the basics of HTML, including how to structure web pages and use semantic elements. In CSS, I’m familiar with core styling concepts and responsive design techniques, such as Flexbox, Grid, and media queries. My JavaScript knowledge includes fundamental concepts like variables, functions, loops, and DOM manipulation, which I use to handle interactivity and dynamic content. I am also capable of building single-page web applications (SPAs), managing basic state, and handling client-side routing.

    Goal:
    I want to become a full stack developer in Django within 8 months."

    """,
        "output": """
    To become a full stack developer in Django within 8 months, you’ll need to build on your current skills by addressing several key areas. While you have a solid foundation in HTML, CSS, JavaScript, and basic SPA development, you will need to focus on learning Django's core concepts, including its MVC architecture, models, views, templates, and URLs. Understanding Django’s ORM for database interactions, implementing user authentication and authorization, and creating RESTful APIs with Django REST Framework are essential for backend development. Additionally, advancing your knowledge in JavaScript frameworks or libraries like React, Angular, or Vue.js will help with front-end development, while learning how to integrate these with Django’s backend is crucial. You’ll also need to become proficient with version control using Git, deployment strategies, and Continuous Integration/Continuous Deployment (CI/CD) practices. Finally, gaining skills in SQL, database design, and testing both backend and frontend components will be important for a comprehensive full stack development capability.
    """,
    }
]

few_shot_skillgap = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=skillgap_examples,
)

skillgap_templ = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a career coah. Given the user's crrent skillset. You need to perform an analysis on user's current skillset and goal to find out what user doesnot know or need to know.
    """,
        ),
        few_shot_skillgap,
        (
            "human",
            """
    Current Skillset:
    {curr_skillset}
    Goal:
    {goal}
    """,
        ),
    ]
)

# Roadmap Generator

roadmap_examples = [
    {
        "input": """
    Current Skillset:
    I know the basics of HTML, including how to structure web pages and use semantic elements. In CSS, I’m familiar with core styling concepts and responsive design techniques, such as Flexbox, Grid, and media queries. My JavaScript knowledge includes fundamental concepts like variables, functions, loops, and DOM manipulation, which I use to handle interactivity and dynamic content. I am also capable of building single-page web applications (SPAs), managing basic state, and handling client-side routing.

    Goal:
    I want to become a full stack developer in Django within 8 months."


    Skillgap:
    To become a full stack developer in Django within 8 months, you’ll need to build on your current skills by addressing several key areas. While you have a solid foundation in HTML, CSS, JavaScript, and basic SPA development, you will need to focus on learning Django's core concepts, including its MVC architecture, models, views, templates, and URLs. Understanding Django’s ORM for database interactions, implementing user authentication and authorization, and creating RESTful APIs with Django REST Framework are essential for backend development. Additionally, advancing your knowledge in JavaScript frameworks or libraries like React, Angular, or Vue.js will help with front-end development, while learning how to integrate these with Django’s backend is crucial. You’ll also need to become proficient with version control using Git, deployment strategies, and Continuous Integration/Continuous Deployment (CI/CD) practices. Finally, gaining skills in SQL, database design, and testing both backend and frontend components will be important for a comprehensive full stack development capability.

    """,
        "output": """
    [
    {"Django Basics": "Start by diving into Django’s foundational concepts. Begin with understanding the Model-View-Template (MVT) architecture, which separates your data model, business logic, and presentation layer. Familiarize yourself with Django’s core components: models for database schema, views for business logic, and templates for rendering HTML. Follow the official Django tutorial to set up a basic project and create simple applications. Build small projects to practice routing URLs, rendering templates, and handling forms. As you progress, learn about Django’s built-in features such as the admin interface, which simplifies managing your data. This foundational knowledge will be critical for building more complex applications later on."},

    {"Django ORM": "Focus on mastering Django’s Object-Relational Mapping (ORM) to manage database interactions. The ORM allows you to define your database schema using Python classes, making it easier to create, retrieve, update, and delete records without writing raw SQL queries. Study how to create and manipulate models, and use Django’s migration system to apply changes to your database schema. Practice writing queries using Django’s query API and explore advanced features such as aggregations and annotations. Understanding how to efficiently use the ORM will enable you to handle data operations effectively and integrate your backend with your frontend."},

    {"Authentication and Authorization": "Implement user authentication and authorization features in your Django projects. Learn how to use Django’s built-in authentication system to handle user registration, login, logout, and password management. Study how to set up user permissions and roles to control access to different parts of your application. Implement custom user models if needed and explore third-party packages for additional functionality, such as social authentication. Ensure that your application can securely manage user sessions and protect sensitive data. Proper authentication and authorization are crucial for building secure applications that handle user data responsibly."},

    {"Django REST Framework (DRF)": "Expand your skills by learning Django REST Framework (DRF) to build RESTful APIs. DRF provides powerful tools for creating and managing APIs, including serializers for converting complex data types into JSON, and viewsets for handling CRUD operations. Study how to set up routers to automate URL routing and implement authentication mechanisms for your APIs. Practice creating endpoints to handle different types of requests and use DRF’s features to ensure your APIs are efficient and secure. Building RESTful APIs with DRF will allow your applications to interact with other systems and support complex front-end interactions."},

    {"JavaScript Frameworks/Libraries": "Advance your JavaScript skills by exploring modern frameworks or libraries like React, Angular, or Vue.js. These tools offer advanced features for building interactive and dynamic user interfaces. Learn about component-based architecture, where the user interface is broken down into reusable components. Study state management techniques to handle complex application states and data flow. Practice integrating these frameworks with APIs to fetch and display data dynamically. By mastering a JavaScript framework, you’ll enhance your ability to create sophisticated, high-performance front-end applications that can interact seamlessly with your Django backend."},

    {"Integrating Frontend with Django": "Develop the skills to integrate your chosen JavaScript framework with Django’s backend. Understand how to connect frontend components with Django APIs, allowing for seamless data exchange between the client and server. Practice setting up AJAX or Fetch API calls to retrieve and send data asynchronously. Work on handling authentication and session management between your frontend and Django backend. Ensure that data is synchronized correctly and that user interactions lead to the expected backend responses. Effective integration is essential for creating cohesive applications where the frontend and backend work together efficiently."},

    {"Version Control with Git": "Become proficient with Git for version control to manage your codebase effectively. Learn how to initialize repositories, create branches, and merge changes. Understand how to commit changes with meaningful messages, resolve merge conflicts, and use Git workflows to collaborate with others. Familiarize yourself with platforms like GitHub or GitLab to host repositories, track issues, and review pull requests. Version control is crucial for managing code changes, collaborating with other developers, and maintaining a history of your project’s evolution. Mastering Git will streamline your development process and improve code quality."},

    {"Deployment Strategies": "Gain knowledge in deploying Django applications to production environments. Study different deployment strategies, including using web servers like Nginx or Apache and application servers like Gunicorn. Learn how to configure and deploy your Django application on cloud platforms such as Heroku, AWS, or DigitalOcean. Practice setting up your environment for production, including configuring static and media file handling, database settings, and security measures like HTTPS. Understanding deployment will help you launch your applications successfully and ensure they are accessible and performant in real-world scenarios."},

    {"CI/CD Practices": "Learn about Continuous Integration and Continuous Deployment (CI/CD) to automate your development workflow. Study how to set up CI/CD pipelines to automatically build, test, and deploy your applications whenever changes are made. Familiarize yourself with tools like Jenkins, Travis CI, or GitHub Actions for creating and managing pipelines. Implement automated testing to catch bugs early and ensure code quality. CI/CD practices will help you maintain a streamlined development process, reduce manual errors, and accelerate the delivery of new features and updates."},

    {"SQL and Database Design": "Enhance your skills in SQL and database design to manage relational databases effectively. Study how to write complex SQL queries for data retrieval, manipulation, and optimization. Understand database schema design principles, including normalization and indexing, to improve database performance and organization. Practice designing and managing relational databases, and learn how to integrate them with Django using the ORM. Proficiency in SQL and database design will enable you to build efficient and scalable applications that handle large volumes of data effectively."}
    ]
    """,
    }
]

few_shot_roadmap = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=roadmap_examples,
)

roadmap_templ = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
      You are a career guide/teacher who specialized in creating a customized or tailored roadmap. Given the user's current skillset, end goal and the skill gap create a personalized roadmap that is easy to follow and understand. The output should be a json array of objects where each object name is topic user should learn and value is the explanation of what he should do. Only include top 10 most relevant topics user should learn. The explanation should be about 200 to 300 words per each topic.
      """,
        ),
        few_shot_roadmap,
        (
            "human",
            """
    Current Skillset:
    {curr_skillset}
    Goal:
    {goal}
    SkillGap:
    {skillgap}
    """,
        ),
    ]
)

# Defining the model

load_dotenv()

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 5000,
}

chat = ChatWatsonx(
    model_id="ibm/granite-13b-instruct-v2",
    project_id="8306ea50-a0cc-4cad-bdd0-d6c2b30623b8",
    params=parameters,
)

skillgap_chain = skillgap_templ | chat | StrOutputParser()
roadmap_chain = roadmap_templ | chat | StrOutputParser()

# goal = input("What is your end goal?\n> ")
# curr_skill = input("What is your current skillset?\n> ")

goal = "I want to become a DevOps with docker and aws skills in 18 months."
curr_skill = "I know APIs and how create a server along with basic linux commands and ecosystem. I also know how to work with github. I know how a server works and HTTP handles request over internet. I know about SQL database and I have been using postgres. I know how to spinup and work with Flask servers in python."

skillgap = skillgap_chain.invoke({"goal": goal, "curr_skillset": curr_skill})

print(skillgap)

roadmap = roadmap_chain.invoke(
    {"goal": goal, "curr_skillset": curr_skill, "skillgap": skillgap}
)

print(roadmap)
