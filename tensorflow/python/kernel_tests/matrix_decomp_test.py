import tensorflow as tf
import numpy as np

def getTestMatrix():
    return np.random.randn(4, 3)

def numpyTestSvd(test_in):
    U, s, V = np.linalg.svd(test_in, full_matrices=True)

def numpyTestSvdS(test_in):
    U, s, V = np.linalg.svd(test_in, full_matrices=True)
    return s

def numpyTestSvdU(test_in):
    U, s, V = np.linalg.svd(test_in, full_matrices=True)
    return U

def numpyTestSvdV(test_in):
    U, s, V = np.linalg.svd(test_in, full_matrices=True)
    return V

def numpyTestQr(test_in):
    return np.linalg.qr(test_in,mode='complete')


class MatrixDecompOpTest(tf.test.TestCase):
    def testSvdS(self):
        a = getTestMatrix()
        print("Test sdv S")
        np_ans = numpyTestSvdS(a)
        print("np_ans:")
        print(np_ans)
        
        with self.test_session():
            tf_ans = tf.matrix_decomp_svd_s(a).eval()
            print("tf_ans")
            print(tf_ans)

        self.assertAllClose(np_ans, tf_ans,atol=1e-5, rtol=1e-5)

    def testSvdU(self):
        a = getTestMatrix()
        print("Test sdv U")
        np_ans = numpyTestSvdU(a)
        print("np_ans:")
        print(np_ans)

        with self.test_session():
            tf_ans = tf.matrix_decomp_svd_u(a).eval()
            print("tf_ans")
            print(tf_ans)

        #self.assertAllClose(np_ans, tf_ans,atol=1e-5, rtol=1e-5)

    def testSvdV(self):
        a = getTestMatrix()
        print("Test sdv V")
        np_ans = numpyTestSvdV(a)
        print("np_ans:")
        print(np_ans)
        
        with self.test_session():
            tf_ans = tf.matrix_decomp_svd_v(a).eval()
            print("tf_ans")
            print(tf_ans)

        #self.assertAllClose(np_ans, tf_ans,atol=1e-5, rtol=1e-5)

    def testSvd(self):
        a = getTestMatrix()
        with self.test_session():
            tf_ans_u = tf.matrix_decomp_svd_u(a).eval()
            tf_ans_s = tf.matrix_decomp_svd_s(a).eval()
            tf_ans_v = tf.matrix_decomp_svd_v(a).eval()

        full_s = np.diag(tf_ans_s)

        np_reconstr = np.dot( np.dot(tf_ans_u,  full_s), np.transpose(tf_ans_v) )
        self.assertAllClose(a, np_reconstr,atol=1e-5, rtol=1e-5)

    def testSvdTransp(self):
        a = np.transpose(getTestMatrix())
        with self.test_session():
            tf_ans_u = tf.matrix_decomp_svd_u(a).eval()
            tf_ans_s = tf.matrix_decomp_svd_s(a).eval()
            tf_ans_v = tf.matrix_decomp_svd_v(a).eval()

        full_s = np.diag(tf_ans_s)

        np_reconstr = np.dot( np.dot(tf_ans_u,  full_s), np.transpose(tf_ans_v) )
        self.assertAllClose(a, np_reconstr,atol=1e-5, rtol=1e-5)
        
    def testQr(self):
        a = getTestMatrix()
        np_q, np_r = numpyTestQr(a)
        
        with self.test_session():
            tf_ans_q = tf.matrix_decomp_qr_q(a).eval()

        print("qr test")
        print("tf:")
        print(tf_ans_q)

        print("np:")
        print(np_q)
        self.assertAllClose(np_q, tf_ans_q,atol=1e-5, rtol=1e-5)
        
if __name__ == '__main__':
    tf.test.main()
