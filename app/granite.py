from langchain_ibm import ChatWatsonx
from dotenv import load_dotenv
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain.schema.runnable import RunnableLambda
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain.schema.output_parser import StrOutputParser
import json

# Example for fewshot prompting

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)


# SkillGap Analysis

skillgap_analysis_examples = [
    {
        "input": """
        Based on your recent completion of an online course in graphic design and your proficiency with Adobe Photoshop, Illustrator, and InDesign, you have developed foundational skills in this field. Your ability to create simple logos, posters, and social media graphics indicates that you have a solid grasp of the basic tools and techniques required for graphic design. However, you are still working on becoming more comfortable with complex designs, which suggests that you are in the early to intermediate stage of your learning journey.

        Your knowledge of Photoshop, Illustrator, and InDesign gives you a versatile skill set to create a wide range of digital and print designs. Photoshop is ideal for photo editing and raster-based designs, Illustrator is your go-to tool for vector-based graphics, and InDesign is used for layout design, especially for print media. Being familiar with these tools equips you to handle diverse projects, though mastering more advanced techniques, such as intricate vector illustrations, advanced photo manipulations, and detailed layout compositions, will be crucial as you progress.

        Your goal of becoming a freelance graphic designer aligns well with your current skill set, especially given your interest in building a portfolio. Taking on small projects is a strategic approach to gaining experience, improving your design skills, and building a client base. As you build your portfolio, focusing on a niche (e.g., branding, social media graphics) could help you stand out to potential clients. Additionally, learning about client management, pricing, and marketing yourself as a freelancer will be essential as you aim to establish a steady stream of work.
        """,
        "output": """
        Your recent completion of an online course in graphic design and your proficiency with tools like Adobe Photoshop, Illustrator, and InDesign indicate that you have built a solid foundation in this field. Your ability to create simple logos, posters, and social media graphics demonstrates that you are familiar with the basic functions and techniques necessary for graphic design. However, since you are still working on becoming more comfortable with complex designs, you are likely in the early to intermediate stages of your learning journey.

        Your knowledge of Photoshop, Illustrator, and InDesign equips you with a versatile skill set for handling both digital and print designs. Photoshop is your primary tool for photo editing and raster-based work, Illustrator is essential for creating vector-based graphics like logos and illustrations, and InDesign is useful for designing layouts, especially for print media. While you are confident in these tools for simpler projects, advancing your skills in more complex design techniques, such as detailed vector illustrations, advanced image manipulation, and intricate layout compositions, will be crucial for further growth.

        Your ambition to become a freelance graphic designer aligns well with your current skill level. Starting with small projects is a practical strategy to build experience and refine your design abilities while also gradually establishing a client base. As you develop your portfolio, focusing on a niche area—such as branding, social media graphics, or editorial design—can help differentiate you in a competitive market. Additionally, gaining knowledge in areas like client communication, project management, pricing strategies, and marketing yourself as a freelancer will be critical as you work towards establishing a steady stream of clients and projects.

        Overall, you have a good foundation in graphic design, with strong potential for growth as you tackle more complex designs and build experience through freelance work. Your approach to starting small and gradually building up your skills and client base is well-aligned with your current stage and long-term goals.
        """,
    },
    {
        "input": """
        Your recent introduction to Python and your familiarity with fundamental programming concepts like variables, loops, and functions lay a solid foundation for moving forward in your data analysis journey. The fact that you’ve also explored libraries like Pandas and NumPy suggests you’ve taken steps toward learning tools essential for data manipulation and numerical analysis, which are crucial in the field of data analytics. These libraries allow you to clean, organize, and analyze data, enabling you to perform operations like filtering, aggregating, and transforming datasets.

        However, understanding how to apply these skills to real-world data analysis projects is key to advancing beyond theoretical knowledge. Transitioning from basic exercises to handling actual datasets, drawing insights, and creating meaningful visualizations will help you develop a more practical understanding. Real-world projects often involve working with messy, unstructured data, dealing with missing values, and identifying patterns that can inform decision-making.

        Your goal of becoming a data analyst aligns well with your current trajectory, but gaining more experience with end-to-end projects—starting from data collection and cleaning to performing analysis and presenting results—is crucial. You might consider starting with small, personal projects where you analyze publicly available datasets (e.g., from Kaggle or government databases) to build your portfolio. As you progress, learning more about data visualization (e.g., using Matplotlib or Seaborn) and getting familiar with SQL for database querying will add more depth to your skill set.

        In summary, you have a good foundation in Python and data manipulation libraries, but gaining hands-on experience with real-world projects and expanding your skill set in areas like data visualization and SQL will be crucial to reaching your goal of working as a data analyst.""",
        "output": """
        Your recent introduction to Python and your familiarity with fundamental programming concepts like variables, loops, and functions lay a solid foundation for moving forward in your data analysis journey. The fact that you’ve also explored libraries like Pandas and NumPy suggests you’ve taken steps toward learning tools essential for data manipulation and numerical analysis, which are crucial in the field of data analytics. These libraries allow you to clean, organize, and analyze data, enabling you to perform operations like filtering, aggregating, and transforming datasets.

        However, understanding how to apply these skills to real-world data analysis projects is key to advancing beyond theoretical knowledge. Transitioning from basic exercises to handling actual datasets, drawing insights, and creating meaningful visualizations will help you develop a more practical understanding. Real-world projects often involve working with messy, unstructured data, dealing with missing values, and identifying patterns that can inform decision-making.

        Your goal of becoming a data analyst aligns well with your current trajectory, but gaining more experience with end-to-end projects—starting from data collection and cleaning to performing analysis and presenting results—is crucial. You might consider starting with small, personal projects where you analyze publicly available datasets (e.g., from Kaggle or government databases) to build your portfolio. As you progress, learning more about data visualization (e.g., using Matplotlib or Seaborn) and getting familiar with SQL for database querying will add more depth to your skill set.

        In summary, you have a good foundation in Python and data manipulation libraries, but gaining hands-on experience with real-world projects and expanding your skill set in areas like data visualization and SQL will be crucial to reaching your goal of working as a data analyst.
        """,
    },
]

few_shot_skillgap = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=skillgap_analysis_examples,
)

skillgap_templ = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Persona: You are an AI assistant specialized in career guidance and skill assessment.

            Task: Analyze the user’s input, which includes their current skill set, knowledge about a particular topic, and their end goal. Identify key areas where the user needs to improve to reach their end goal. Focus on the most critical skill gaps and keep the output concise for website display, using the third person.

            Format: The output should be 2-3 sentences in plain text without any markdowns or bullet points
            """,
        ),
        few_shot_skillgap,
        ("human", "{input}"),
    ]
)


# Roadmap Generator

roadmap_examples = [
    {
        "input": """
        Your recent completion of an online course in graphic design and your proficiency with tools like Adobe Photoshop, Illustrator, and InDesign indicate that you have built a solid foundation in this field. Your ability to create simple logos, posters, and social media graphics demonstrates that you are familiar with the basic functions and techniques necessary for graphic design. However, since you are still working on becoming more comfortable with complex designs, you are likely in the early to intermediate stages of your learning journey.

        Your knowledge of Photoshop, Illustrator, and InDesign equips you with a versatile skill set for handling both digital and print designs. Photoshop is your primary tool for photo editing and raster-based work, Illustrator is essential for creating vector-based graphics like logos and illustrations, and InDesign is useful for designing layouts, especially for print media. While you are confident in these tools for simpler projects, advancing your skills in more complex design techniques, such as detailed vector illustrations, advanced image manipulation, and intricate layout compositions, will be crucial for further growth.

        Your ambition to become a freelance graphic designer aligns well with your current skill level. Starting with small projects is a practical strategy to build experience and refine your design abilities while also gradually establishing a client base. As you develop your portfolio, focusing on a niche area—such as branding, social media graphics, or editorial design—can help differentiate you in a competitive market. Additionally, gaining knowledge in areas like client communication, project management, pricing strategies, and marketing yourself as a freelancer will be critical as you work towards establishing a steady stream of clients and projects.

        Overall, you have a good foundation in graphic design, with strong potential for growth as you tackle more complex designs and build experience through freelance work. Your approach to starting small and gradually building up your skills and client base is well-aligned with your current stage and long-term goals.
        """,
        "output": """
        [
          {
              "Advanced Photoshop Techniques": "To elevate your graphic design skills, delve into advanced Photoshop techniques. This includes mastering layer styles, blending modes, and complex selection tools. Learn about non-destructive editing practices such as using adjustment layers and layer masks. Explore techniques for creating sophisticated composites, advanced retouching, and color correction. These skills will help you handle more intricate design tasks, such as high-end photo manipulations and detailed digital art, broadening your capabilities and enhancing your design portfolio."
          },
          {
              "Illustrator Vector Mastery": "Deepen your understanding of Adobe Illustrator by focusing on advanced vector techniques. Study how to create detailed vector illustrations, complex shapes, and custom brushes. Learn about pathfinder tools, gradients, and mesh tools to produce intricate and scalable designs. Mastering these skills will allow you to create high-quality logos, icons, and illustrations that maintain their clarity and precision across various sizes and applications, enhancing your ability to work on diverse design projects."
          },
          {
              "InDesign Layout Design": "Improve your proficiency with Adobe InDesign by mastering advanced layout design techniques. Learn how to create multi-page documents, utilize advanced typography settings, and manage styles efficiently. Study how to work with grids, guides, and master pages to produce professional-quality print materials like brochures, magazines, and annual reports. Understanding how to effectively design layouts will enhance your ability to produce polished and cohesive print and digital publications."
          },
          {
              "Freelance Business Skills": "Develop essential freelance business skills to succeed in the competitive graphic design market. Study client communication techniques, including how to effectively present and pitch your work. Learn project management skills to handle multiple projects and deadlines efficiently. Understand pricing strategies and contracts to ensure fair compensation and protect your work. Focus on marketing yourself through online portfolios, social media, and networking to attract and retain clients. These skills are crucial for establishing and growing a successful freelance graphic design career."
          },
          {
              "Design Portfolio Development": "Build a strong design portfolio that showcases your best work and demonstrates your range of skills. Focus on curating projects that highlight your expertise in various design areas, such as branding, social media graphics, and print materials. Include detailed case studies for each project, describing your design process and the challenges you overcame. A well-crafted portfolio will help you attract potential clients and employers, making it easier to establish yourself as a proficient and versatile graphic designer."
          },
          {
              "Client Communication and Management": "Enhance your ability to communicate and manage client relationships effectively. Learn how to set clear expectations, provide regular updates, and handle feedback professionally. Study how to negotiate project scopes and timelines to align with client needs. Implement strategies for managing client revisions and ensuring satisfaction throughout the project lifecycle. Strong client communication and management skills are vital for building long-term relationships and maintaining a positive reputation in the freelance market."
          },
          {
              "Advanced Typography": "Expand your knowledge of typography to improve your design work. Study advanced typography principles, including kerning, tracking, and leading. Explore the use of typographic hierarchy to enhance readability and visual appeal. Learn how to select and pair fonts effectively to complement your designs and convey the intended message. Mastering typography will help you create more visually compelling and professional designs, enhancing your overall design quality."
          },
          {
              "Web Design Fundamentals": "Gain an understanding of web design principles to expand your design capabilities. Study basic HTML and CSS to understand how designs are implemented on the web. Learn about responsive design techniques to ensure your designs look great on various devices and screen sizes. Familiarize yourself with web design tools like Adobe XD or Figma for creating and prototyping web layouts. This knowledge will help you transition from print to digital design, broadening your skill set and marketability."
          },
          {
              "Design Trends and Innovation": "Stay updated on current design trends and innovations to keep your work fresh and relevant. Follow industry blogs, attend design conferences, and participate in design communities to learn about emerging trends, new tools, and creative techniques. Understanding current trends will help you create modern and appealing designs that resonate with clients and audiences. Continuously evolving your design approach will ensure that your work remains innovative and competitive in the ever-changing design landscape."
          }
      ]
      """,
    },
    {
        "input": """
        Your recent introduction to Python and your familiarity with fundamental programming concepts like variables, loops, and functions lay a solid foundation for moving forward in your data analysis journey. The fact that you’ve also explored libraries like Pandas and NumPy suggests you’ve taken steps toward learning tools essential for data manipulation and numerical analysis, which are crucial in the field of data analytics. These libraries allow you to clean, organize, and analyze data, enabling you to perform operations like filtering, aggregating, and transforming datasets.

        However, understanding how to apply these skills to real-world data analysis projects is key to advancing beyond theoretical knowledge. Transitioning from basic exercises to handling actual datasets, drawing insights, and creating meaningful visualizations will help you develop a more practical understanding. Real-world projects often involve working with messy, unstructured data, dealing with missing values, and identifying patterns that can inform decision-making.

        Your goal of becoming a data analyst aligns well with your current trajectory, but gaining more experience with end-to-end projects—starting from data collection and cleaning to performing analysis and presenting results—is crucial. You might consider starting with small, personal projects where you analyze publicly available datasets (e.g., from Kaggle or government databases) to build your portfolio. As you progress, learning more about data visualization (e.g., using Matplotlib or Seaborn) and getting familiar with SQL for database querying will add more depth to your skill set.

        In summary, you have a good foundation in Python and data manipulation libraries, but gaining hands-on experience with real-world projects and expanding your skill set in areas like data visualization and SQL will be crucial to reaching your goal of working as a data analyst.
        """,
        "output": """
        [
            {
                "Real-World Data Analysis Projects": "To bridge the gap between theoretical knowledge and practical application, focus on undertaking real-world data analysis projects. Start by selecting publicly available datasets from platforms like Kaggle or government databases. Work on tasks such as data cleaning, exploratory data analysis (EDA), and deriving insights from the data. These projects will help you understand the challenges of working with messy, unstructured data, handle missing values, and identify meaningful patterns. Document your process and findings to build a portfolio that showcases your ability to tackle real data problems and draw actionable conclusions."
            },
            {
                "Data Visualization with Matplotlib and Seaborn": "Enhance your data analysis skills by learning how to create compelling visualizations using libraries like Matplotlib and Seaborn. These tools allow you to visualize data distributions, trends, and relationships effectively. Master techniques for creating various types of plots, such as histograms, scatter plots, and heatmaps. Understanding how to present data visually will help you communicate insights more clearly and make your analyses more impactful. Developing proficiency in data visualization is crucial for interpreting complex datasets and conveying results to stakeholders in an accessible manner."
            },
            {
                "SQL for Data Querying": "Gain proficiency in SQL (Structured Query Language) to manage and query relational databases efficiently. SQL is essential for extracting, filtering, and aggregating data from databases. Learn how to write complex queries to join tables, perform subqueries, and aggregate data. Understanding SQL will enable you to interact with large datasets and retrieve relevant information for analysis. Combining SQL skills with Python data manipulation will enhance your ability to perform comprehensive data analysis and work with a wide range of data sources."
            },
            {
                "Advanced Pandas Techniques": "Deepen your knowledge of Pandas by exploring advanced techniques for data manipulation and analysis. Study how to handle time series data, work with multi-index DataFrames, and perform advanced grouping and aggregation operations. Learn about efficient data handling methods, such as using vectorized operations and optimizing performance with large datasets. Mastering these techniques will enable you to manage more complex data structures and perform sophisticated analyses, improving your overall efficiency and effectiveness as a data analyst."
            },
            {
                "Data Cleaning and Preprocessing": "Focus on mastering data cleaning and preprocessing techniques to prepare datasets for analysis. Learn how to handle missing values, detect and correct errors, and standardize data formats. Study methods for data normalization and transformation to ensure that your datasets are clean and ready for analysis. Effective data cleaning is crucial for accurate and reliable analysis, as it ensures that the data you work with is high quality and suitable for generating meaningful insights."
            },
            {
                "Exploratory Data Analysis (EDA)": "Develop your skills in Exploratory Data Analysis (EDA) to better understand and summarize datasets. EDA involves using statistical and graphical techniques to uncover patterns, detect anomalies, and test hypotheses. Learn how to generate descriptive statistics, visualize distributions, and explore relationships between variables. EDA helps in forming hypotheses and guiding further analysis, making it an essential step in the data analysis workflow. Proficiency in EDA will enhance your ability to make data-driven decisions and inform your analytical approach."
            },
            {
                "Machine Learning Basics": "Expand your skill set by learning the basics of machine learning, which can enhance your data analysis capabilities. Study fundamental concepts such as supervised and unsupervised learning, and familiarize yourself with algorithms like linear regression, decision trees, and clustering. Learn how to use libraries like Scikit-learn for building and evaluating machine learning models. Understanding these concepts will enable you to apply predictive analytics and classification techniques to your data analysis projects, adding an advanced dimension to your analytical skills."
            },
            {
                "Project Management and Documentation": "Develop skills in project management and documentation to effectively manage data analysis projects from start to finish. Learn how to define project goals, create project plans, and track progress. Practice documenting your analysis process, including data sources, methodologies, and findings. Effective project management ensures that you stay organized and meet deadlines, while thorough documentation makes it easier to communicate your work and results to others. These skills are essential for managing complex projects and presenting your analysis clearly and professionally."
            },
            {
                "Continuous Learning and Skill Development": "Stay updated with the latest trends and advancements in data analysis and data science. Follow industry blogs, take online courses, and participate in data analysis communities to continually improve your skills and knowledge. Explore new tools and technologies as they emerge, and apply them to your projects to stay competitive in the field. Continuous learning is crucial for adapting to changes in the industry and maintaining a high level of expertise in data analysis."
            }
        ]
      """,
    },
]

few_shot_roadmap = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=roadmap_examples,
)

# Modified roadmap_chain to output data suitable for graph visualization
roadmap_templ = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Persona: You are an AI assistant specialized in career guidance and skill assessment. Your role is to evaluate the user's current skill set, knowledge on a specific topic, and end goal, providing a detailed analysis.

            Task: Analyze the user’s input, which includes their current skill set, knowledge about a particular topic, and their end goal along with skill gap analysis between the end goal and current position.

            Your output should be a JSON array where each object represents a step in the roadmap, focusing on a clear step-wise progression. Each object should have the following keys:

            * "step": The numerical step in the roadmap
            * "skill": The name of the skill or technology to be learned in this step
            * "description": A brief description of the skill and its relevance to the end goal
            * "resources": (Optional) An array of recommended learning resources for this skill (e.g., online courses, books, projects)
            """,
        ),
        few_shot_roadmap,
        ("human", "{input}"),
    ]
)


# Defining the model

load_dotenv()

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 8000,
}

chat = ChatWatsonx(
    model_id="ibm/granite-13b-chat-v2",
    project_id="8306ea50-a0cc-4cad-bdd0-d6c2b30623b8",
    params=parameters,
)


def json_decode(json_str):
    start_index = json_str.find("[")
    end_index = json_str.rfind("]")

    trimmed_json_str = json_str[start_index : end_index + 1]

    try:
        return json.loads(trimmed_json_str)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return {}


decoder = RunnableLambda(lambda x: json_decode(x))

skillgap_chain = skillgap_templ | chat | StrOutputParser()


roadmap_chain = roadmap_templ | chat | StrOutputParser() | decoder


def get_roadmaps(curr_skillset) -> dict:
    skillgap = skillgap_chain.invoke({"input": curr_skillset})
    roadmap = roadmap_chain.invoke({"input": skillgap})
    return roadmap
