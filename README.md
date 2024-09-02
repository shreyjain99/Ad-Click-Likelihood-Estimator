<h2 align= "center"><em>Ad Click Likelihood Estimator</em></h2>

<div align="center">
  <img height="400" src="https://github.com/shreyjain99/Ad-Click-Likelihood-Estimator/blob/main/src%20files/cover%20image.webp"/>
</div>

<hr width="100%" size="2">

<h3 align= "left"> <b> Key Project Formulation </b> </h3>

<br>

<p>
<strong>Business Objective :</strong> To predict the probability of click and maximize profit of the ad while keeping the search results relevant for the product ad.
</p>

<br>

<p>
<strong>Constraints :</strong>
</p>
<ol>
<li>Latency (probably in milliseconds) </li>
<li>Interpretability (just for sanity check)</li>
<li>parallelizable training</li>
</ol>

<br>

<p>
<strong>Get the data from :</strong> https://www.kaggle.com/c/kddcup2012-track2
</p>

<br>

<p>
<strong>Data Collection :</strong>
We Have collected all yellow taxi trips data of jan-2015 to mar-2015 and jan-2016 to mar-2016
</p>
<table style="width:50%;text-align:center;">
<caption style="text-align:center;">Data Files</caption>
<tr>
<td><b>Filename</b></td><td><b>Available Format</b></td>
</tr>
<tr>
<td>training</td><td>.txt (9.9Gb)</td>
</tr>
<tr>
<td>queryid_tokensid</td><td>.txt (704Mb)</td>
</tr>
<tr>
<td>purchasedkeywordid_tokensid</td><td>.txt (26Mb)</td>
</tr>
<tr>
<td>titleid_tokensid</td><td>.txt (172Mb)</td>
</tr>
<tr>
<td>descriptionid_tokensid</td><td>.txt (268Mb)</td>
</tr>
<tr>
<td>userid_profile</td><td>.txt (284Mb)</td>
</tr>
</table>



<br>

<p>
<strong>Features in the dataset :</strong>
</p>
<table style="width:100%">
  <caption style="text-align:center;">training.txt</caption>
  <tr>
    <th>Feature</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>UserID</td>
    <td>The unique id for each user</td>
    </tr>
  <tr>
    <td>AdID</td>
    <td>The unique id for each ad</td>
  </tr>
  <tr>
    <td>QueryID</td>
    <td>The unique id for each Query (it is a primary key in Query table(queryid_tokensid.txt))</td>
  </tr>
  <tr>
    <td>Depth</td>
    <td>The number of ads impressed in a session is known as the 'depth'. </td>
  </tr>
  <tr>
    <td>Position</td>
    <td>The order of an ad in the impression list is known as the ‘position’ of that ad.</td>
  </tr>
  <tr>
    <td>Impression</td>
    <td>The number of search sessions in which the ad (AdID) was impressed by the user (UserID) who issued the query (Query).</td>
  </tr>
  <tr>
    <td>Click</td>
    <td>The number of times, among the above impressions, the user (UserID) clicked the ad (AdID).</td>
  </tr>
  <tr>
    <td>TitleId</td>
    <td>A property of ads. This is the key of 'titleid_tokensid.txt'. [An Ad, when impressed, would be displayed as a short text known as ’title’, followed by a slightly longer text known as the ’description’, and a URL (usually shortened to save screen space) known as ’display URL’.]</td>
  </tr>
  <tr>
    <td>DescId</td>
    <td>A property of ads.  This is the key of 'descriptionid_tokensid.txt'. [An Ad, when impressed, would be displayed as a short text known as ’title’, followed by a slightly longer text known as the ’description’, and a URL (usually shortened to save screen space) known as ’display URL’.]</td>
  </tr>
  <tr>
    <td>AdURL</td>
    <td>The URL is shown together with the title and description of an ad. It is usually the shortened landing page URL of the ad, but not always. In the data file,  this URL is hashed for anonymity.</td>
  </tr>
  <tr>
    <td>KeyId</td>
    <td>A property of ads. This is the key of  'purchasedkeyword_tokensid.txt'.</td>
  </tr>
  <tr>
    <td>AdvId</td>
    <td>a property of the ad. Some advertisers consistently optimize their ads, so the title and description of their ads are more attractive than those of others’ ads.</td>
  </tr>
</table>

<br>

<p>
<strong>ML Problem Formulation :</strong>
</p>
<p><b><u> Time-series forecasting and Regression</u></b></p>
-<i> To find number of pickups, given location cordinates(latitude and longitude) and time, in the query reigion and surrounding regions therefore we predict number of pickups accurately as possible for each region in a 10 min interval.</i> (we would be using data collected in Jan - Mar 2015 to predict the pickups in Jan - Mar 2016.)

<br>
<br>

<p>
<strong>Performance metrics :</strong>
</p>
<ol>
<li>Mean Absolute percentage error</li>
<li>Mean Squared error</li>
</ol>

<hr width="100%" size="2">

<br>

<body>

  <h3>Summary</h3>

  <h4>Data Processing</h4>
    <p>The project begins with data cleaning and the removal of outliers through univariate analysis of key features. This step ensures that the data used for modeling is both accurate and relevant.</p>

  <h4>Clustering and Region Division</h4>
    <p>Following data cleaning, the city is divided into 40 regions based on clustering techniques that consider both distance and time intervals. This division allows for a more granular and accurate prediction of taxi demand across the city.</p>

  <h4>Prediction Framework</h4>
    <p>For each region, the data is broken down into 10-minute time intervals, and the number of pickups within each interval is predicted. The modeling process begins with baseline models, including moving averages, moving weighted averages, and exponential averages, which serve as simple predictors of demand.</p>

   <h4>Advanced Modeling</h4>
    <p>The project then advances to more sophisticated machine learning models, including linear regression, Random Forest regressor, and XGBoost regressor. These models are employed to enhance prediction accuracy by leveraging the complex relationships in the data.</p>

  <p>The final output of the project is a robust predictive model that can help taxi drivers optimize their routes and improve their earnings by forecasting the most promising areas for pickups in real-time.</p>

</body>

<hr width="100%" size="2">
<br>

<p>
<strong>Future Scope :</strong>
</p>
<ol>
<li>Incorporate fourier transform features as features into regression models </li>
<li>Perform hyperparameter tuning for regression models</li>
<li>Try more regression models as well as artificial neural nets (ann)</li>
</ol>

<hr width="100%" size="2">
