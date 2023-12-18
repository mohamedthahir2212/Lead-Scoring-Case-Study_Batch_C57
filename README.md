# Lead-Scoring-Case-Study_Batch_C57

##LeadScore_assignment:

XEducation is an education organization that sells online courses to industry experts. On any given day, a large number of professionals who are interested in the courses visit their website and search for courses.

The company promotes its courses on various websites and search engines such as Google. When these users arrive at the website, they may browse the courses, fill out a course registration form, or watch some videos. These persons are regarded as leads when they fill out a form with their email address or phone number. Furthermore, the organization receives leads from previous referrals. Once these leads are obtained, members of the sales team begin making calls, composing emails, and so on. Some leads are converted throughout this procedure, whereas the majority are not. At X schooling, the average lead conversion rate is roughly 30%.

X Education now receives a large number of leads, but its lead conversion rate is very low. For example, if they get 100 leads in a day, only around 30 of them will be converted. The organization wants to discover the most potential leads, also known as 'Hot Leads,' to make this process more efficient. If they are successful in identifying this group of leads, the lead conversion rate should increase because the sales staff will now be focusing on connecting with the potential leads rather than calling everyone. The following funnel represents a typical lead conversion process:

Lead Conversion Process - Shown as a funnel Lead Conversion Process - Shown as a funnel As you can see, there are a lot of leads created in the first stage (top), but only a few of them become paying clients in the second step. To acquire a greater lead conversion in the middle stage, you must nurture the potential leads well (e.g., educating the leads about the product, regularly communicating, etc.).

X Education has asked you to assist them in identifying the most promising leads, or those who are most likely to convert into paying clients. The company expects you to create a model in which you give a lead score to each lead so that customers with higher lead scores have a higher conversion chance and customers with lower lead scores have a lower conversion chance. The CEO, in particular, has stated that the objective lead conversion rate should be about 80%.

Data You were given a leads dataset from the past with approximately 9000 data points. This dataset contains numerous parameters such as Lead Source, Total Time Spent on Website, Total Visits, Last Activity, and so on that may or may not be relevant in determining whether or not a lead will be converted. In this situation, the target variable is the column 'Converted,' which indicates whether a previous lead was converted or not, with 1 indicating that it was converted and 0 indicating that it was not converted. The data dictionary included in the zip folder at the bottom of the page can help you learn more about the dataset.

Another thing to look for are the levels in the category variables. Many categorical variables have a level named 'Select' that must be handled because it is equivalent to a null value (consider why).

Case Study Objectives: This case study has several objectives

Create a logistic regression model to assign a lead score between 0 and 100 to each lead, which the organization may use to target potential leads. A greater number indicates that the lead is hot, i.e. probable to convert, whereas a lower value indicates that the lead is cool and unlikely to convert. There are certain more issues raised by the firm that your model should be able to handle if the company's requirements alter in the future, so you will need to address these as well. These issues are contained in a separate doc file. 

Please complete it using the logistic regression model you obtained in the first step. Also, make sure to include this in your final PowerPoint presentation where you'll make recommendations.
