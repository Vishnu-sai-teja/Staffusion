## Attention Mechanism Mathematics
- Similatiy between words
    * Dot product
    * Cosine similarity
- Key , Query , Value matrices as linear transformations


## Embeddings
* Words or longer peice of words , that are learnt to be put together
    - Examples , fruits are put together , electronics together
    - Words that are close , gets pulled together and that are not similar gets pulled away
    - For example we are having a  bunch of fruits alread in it , then it would be able to identify when another fruit and comes together

## Similarity
* Words are measured with similarity between them
    * When words are similar the similarity is high (orange and bananas)
    * When words are not so similar , we get low similarit (oranges and laptop)
* **How To Measure this Similarity**
    * There are three ways to measure the similairty
    * **Dot Porduct**
    	* We consider a vector to represent these words , with each value in the word representing a feature (Kind of tech or a fruitness) 
    	* When we look at 2 fruits the value of fruitness is high for both and the dor product results in a higher or a larger value
    	* In case of the fruit and phone , one has a higher tech value and the other has a higher fruitness value , which results in a lower value of dot product
    	* So things with high dot product are relatable and with low dot product are not so relatable , could be negative too
	*  **Cosine Similarity**
		*  In this case , we consider the angle between the word vector , from the origin
		*  The larger the value of ange the two words make w.r.t the origin , the more they are **unsimilar**
		*  In case of smaller angle , that means the words lie close together and **highly relatable**
	* **Scaled Dot Product**
		* This is what we use in the case of **Attention**
		* We calculate the dot product between the two word vectors , and then we divide them with the value of **root(length of the vector)**
		* So the idea is same as that of the dot product 
		* We divide it with the value of **(root of length of features)** , to make the numbers smaller , easy for computation
	* Example for **cosine Similaity**
		* The cosisne similairt between each word and itself is always 1 , as they make **0 degres**s with themselves
		* We are using this table (we compte similaity between all the words) , and help us out for further words too .
		*  We add the words with ratios of their weightage ,and then we divide them with the sum of the ratios to **Normalize**  the wording with the ratio of the context coming from each of the word.
			*  To avoid dividing the **0** in the denominator , instead of taking the context from each word take the **Exponential Context** form each word , which avoids divide by zero constraint (called as **Soft-Max**)
			*  

