# COMMAND ----------

displayHTML(html_run_search_1)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 2. `run_summary` Stage Attributes
# MAGIC
# MAGIC We'll look at the run_summary method before we look at what needs to be done to run it asynchronously with run_augment. Recall that the output of our previous stage will be a dataclass, but ultimately that dataclass will need to be transformed into it's essential parts as a list of `tuple(id, content)`. Each iteration of run_summary will therefore have content and id variables in context from a single search result.
# MAGIC
# MAGIC <!--  -->
# MAGIC
# MAGIC ![new_schedule](../Includes/images/compound-ai-system-s-2.png)

# COMMAND ----------

displayHTML(html_run_search_2)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 3. (Optional) `run_augment` Stage Attributes
# MAGIC
# MAGIC Now let's plan what's necessary to execute `run_summary` asynchronously. The actual lesson of how asyncio works is beyond the scope of this instruction. However, we can still go through the planning exercise for a stage that runs asynchronously. If you are uncomfortable that you are planning without understanding exactly how every library works, take comfort in the fact that a lot of library discovery takes place during planning.
# MAGIC
# MAGIC
# MAGIC <!--  -->
# MAGIC
# MAGIC ![new_schedule](../Includes/images/compound-ai-system-s-3.png)

# COMMAND ----------

displayHTML(html_run_search_3)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 4. `run_get_context` Stage Attributes
# MAGIC
# MAGIC This stage we just need to take the results of our `run_augment` stage and convert into a context for our QA stage. That's right, a stage doesn't have to include an LLM. While this stage could have been made part of the stage before or after, it's broken out as it's own entity so this compound app could have a non-LLM stage.
# MAGIC
# MAGIC ![new_schedule](../Includes/images/compound-ai-system-s-1.png)
# MAGIC <!--  -->

# COMMAND ----------

displayHTML(html_run_search_4)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 5. `run_qa` Stage Attributes
# MAGIC
# MAGIC We'll look at the `run_qa` stage. To make things interesting we are going to use a Chat LLM type instead of a Completion LLM type like we used in run_summary. That means you should take a look at [Chat Model](https://docs.databricks.com/en/machine-learning/model-serving/score-foundation-models.html#query-a-chat-completion-model) Docs. It may look the same as the completion model signature, but look closely, it's not.
# MAGIC
# MAGIC <!--  -->
# MAGIC
# MAGIC ![new_schedule](../Includes/images/compound-ai-system-s-5.png)

# COMMAND ----------

displayHTML(html_run_search_5)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 6. (Optional) `main` Putting the Stages Together
# MAGIC
# MAGIC Time to take all that work planning each individual stage and putting it all together. Take a look below how our individual state planning greatly simplifies chaining our stages together. Not only is it succinct and easy to read, the logical separation and inclusion of dataclasses will also make the following future developer tasks much more straight forward:
# MAGIC
# MAGIC * **Testing**: Test can be written by each module we've created making it easier than would have otherwise been if we wrote our code as a script.
# MAGIC
# MAGIC * **Tracing**: Beyond the scope of this instruction, but adding trace decorators each stage will allow better evaluation for future improvements.
# MAGIC
# MAGIC * **Parameterizing**: By extraction out what all parameters are in the solution, we now have an understanding of our parameter space and ability to create new model versions by simply modifying parameters with no code updates.
# MAGIC
# MAGIC <!--  -->
# MAGIC
# MAGIC ![new_schedule](../Includes/images/compound-ai-system-s-6.png)

# COMMAND ----------

displayHTML(html_run_search_6)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 7. (Optional) Compound App as Pyfunc Model
# MAGIC
# MAGIC Our main method is nice, it can take a string and return a string. However, while local execution is nice, we'd really like our model to be deployable and be evaluated just as if it were a foundation model or external model. MLFlow and Databricks accomplish model deployments through built-in model types. Since we are not using a model library, we can use MLFlow's general form of a `pyfunc` model, and we can use a specific subclass of `mlflow.PythonModel`.
# MAGIC

# COMMAND ----------

displayHTML(html_run_search_7)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC %md
# MAGIC
# MAGIC ### Full Multi-Endpoint Architecture
# MAGIC
# MAGIC
# MAGIC We've gone through all the work of identifying the dependencies which include both a Data Serving Endpoint and a couple model serving endpoints. We should have a look at what our final architecture is. Even in this straight forward compound application, you can see that it has a lot of endpoint dependencies. It's worth having this perspective to see all the serving endpoints that must be maintained.
# MAGIC
# MAGIC <!--  -->
# MAGIC
# MAGIC ![new_schedule](../Includes/images/compound-ai-system-s-10.png)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demo, we planned a sample compound AI system using pure code. This demo showed how different components can be defined independently and then are linked together to build the system.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="blank">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy" target="blank">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use" target="blank">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/" target="blank">Support</a>
