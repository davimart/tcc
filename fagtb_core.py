from sklearn.tree import DecisionTreeRegressor
import numpy as np
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
from sklearn.metrics import accuracy_score
import math


def lossgr(y, p):
    # Avoid division by zero
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return - y * np.log(p) - (1 - y) * np.log(1 - p)

def p_rule(y_pred, z_values, threshold=0.5):
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
    odds = y_z_1.mean() / y_z_0.mean()
    return np.min([odds, 1/odds]) * 100


class FAGTB(object):

    def __init__(self, n_estimators, learning_rate, min_samples_split,
                 min_impurity, max_depth, max_features, regression):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.regression = regression
        self.max_features = max_features

        # Initialize regression trees
        self.trees = []
        self.clfs = []
        self.lossfunction_adv =[]
        self.losstraining =[]

        for _ in range(n_estimators):
            tree = DecisionTreeRegressor(criterion='friedman_mse', max_depth=9,
  max_features=self.max_features, max_leaf_nodes=None,
  min_impurity_decrease=0.0,
  min_samples_leaf=1, min_samples_split=2,
  min_weight_fraction_leaf=0.0
  , random_state=0)
            self.trees.append(tree)
            clf = LogisticRegression()           
            self.clfs.append(clf)
            self.model = []
    def fit2(self, X, y, sensitive, LAMBDA):
        clf = LogisticRegression()
        clf._initialize_parameters(sensitive)
        print(clf.param)

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)

    def fit(self, X, y, sensitive, LAMBDA, Xtest, yt, sensitivet):

        y2 = np.expand_dims(sensitive, axis=1)

        lfadv =0

        self.Init = np.log(np.sum(y)/np.sum(1-y))
        
        y_pred2 = np.full(np.shape(y), self.Init)
        y_pred = np.full(np.shape(y), self.Init)
        y_predt = np.full(np.shape(yt), self.Init)
        t =np.full(np.shape(y), 0)
        t2 =np.full(np.shape(yt), 0)
        self.LAMBDA = LAMBDA
        proj = 0
        table = [0,0,0,0]
        y_pred2 = np.expand_dims(1/(1+np.exp(-y_pred)), axis=1)

        graph = tf.Graph()
        seed = 7 # for reproducible purpose
        input_size =  1 # number of features

        learning_rate2 = 0.01
        with graph.as_default():

            X_input = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name='X_input')
            y_input = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_input')
            
            W1 = tf.Variable(tf.random_normal(shape=[input_size, 1], seed=seed), name='W1', trainable=True)
            b1 = tf.Variable(tf.random_normal(shape=[1], seed=seed), name='b1', trainable=True)
            sigm = tf.nn.sigmoid(tf.add(tf.matmul(X_input, W1), b1), name='pred')
            logit = tf.add(tf.matmul(X_input, W1), b1)
            loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_input,
                                                                    logits=logit, name='loss'))
            train_steps = tf.train.GradientDescentOptimizer(learning_rate2).minimize(loss)
        
            sigm2 = tf.cast(sigm, tf.float32, name='sigm2') # 1 if >= 0.5
            pred = tf.cast(tf.greater_equal(sigm, 0.5), tf.float32, name='pred') # 1 if >= 0.5
            acc = tf.reduce_mean(tf.cast(tf.equal(pred, y_input), tf.float32), name='acc')
            
            init_var = tf.global_variables_initializer()
            var_grad = tf.gradients(loss, X_input)[0]
            
        train_feed_dict = {X_input: y_pred2, y_input: y2}

        sess = tf.Session(graph=graph)
        sess.run(init_var)
        

        for i in range(self.n_estimators):

            y_pred2 = np.expand_dims(1/(1+np.exp(-y_pred)), axis=1)

            train_feed_dict = {X_input: y_pred2, y_input: y2}   
            sess.run(train_steps, feed_dict=train_feed_dict)
            cur_loss = sess.run(loss, feed_dict=train_feed_dict)
            train_acc = sess.run(acc, feed_dict=train_feed_dict)
            S_ADV = sess.run(sigm2, feed_dict=train_feed_dict)

            gradient_adv = sess.run(var_grad, feed_dict=train_feed_dict)
                            
            if abs(np.sum(gradient_adv)) <0.001 :
                 print('erreur de gradient')

            lfadv = gradient_adv*y_pred2*(1-y_pred2)    # *len(gradient_adv)       

            t=-np.squeeze(lfadv.T)
            proj = 0
            gradient = y- 1/(1+np.exp(-y_pred))- LAMBDA*t -proj
            self.trees[i].fit(X, gradient)
            update = self.trees[i].predict(X)
 
            y_pred += np.multiply(self.learning_rate, update)
            y_fin = 1/(1+np.exp(-y_pred))

            losstraining = lossgr(y,y_fin)
            lossglobal = losstraining - LAMBDA*t

            updatet = self.trees[i].predict(Xtest)
            y_predt += np.multiply(self.learning_rate, updatet) 
            y_predt2=1/(1+np.exp(-y_predt))
            accuracy = accuracy_score(y, np.squeeze(y_fin)>0.5)
            accuracyt = accuracy_score(yt, np.squeeze(y_predt2)>0.5)

            if i % 5 == 0:
                print (i,np.sum(lfadv),np.sum(losstraining),np.sum(lossglobal), "Accuracy:", round(accuracy,4), " test : ", round(accuracyt,4), " Prule Train : ", p_rule(y_fin, sensitive)/100," Prule test : ", p_rule(y_predt2, sensitivet)/100)
            table = np.vstack([table,[accuracy,accuracyt, p_rule(y_fin, sensitive)/100, p_rule(y_predt2, sensitivet)/100]])
        return {'y_pred2':y_pred2,'S_ADV':S_ADV}

    
    #### ====== Mudanças para o predict ====== ####
    def predict_score(self, X):

        y_pred = np.full(np.shape(X)[0], self.Init)

        for i in range(self.n_estimators):
            update = self.trees[i].predict(X)
            y_pred += np.multiply(self.learning_rate, update)

        y_score = 1 / (1 + np.exp(-y_pred))

        return y_score


    def predict_proba(self, X):

        y_score = self.predict_score(X)

        return np.column_stack([
            1 - y_score,
            y_score
        ])


    def predict(self, X):

        y_score = self.predict_score(X)

        return (y_score >= 0.5).astype(int)
    
class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))
   

class LogisticRegression():
    def __init__(self, learning_rate=.1):
        self.param = None
        self.learning_rate = learning_rate
        self.sigmoid = Sigmoid()

    def _initialize_parameters(self, X):
        n_features = np.shape(X)[1]
        limit = 1 / math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))
   
    def fit(self, X, y, iteration):
        y_pred = self.sigmoid(X.dot(self.param))
        self.param -= self.learning_rate * -(y - y_pred).dot(X)
        return self.param

    def gradient_adv(self,X,y):
        y_pred = self.sigmoid(X.dot(self.param))
        gradient_adv = (y - y_pred)*self.param*X.T*(1-X).T
        return gradient_adv

    def predict(self, X):
        y_pred = np.round(self.sigmoid(X.dot(self.param))).astype(int)
        return y_pred

    def lossfunction(self,X,y):
        y_pred = self.sigmoid(X.dot(self.param))
        return lossgr(y,y_pred)
    
    def lossfunction_adv(self,X,y):
        y_pred = self.sigmoid(X.dot(self.param))
        return y-y_pred

    def param2(self):
        return 2*self.param