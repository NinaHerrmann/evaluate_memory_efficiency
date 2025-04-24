import os
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.cluster import KMeans
import numpy as np
import subprocess
import re
import random
import uuid
import shutil
import tempfile

@dataclass
class Tree:
    """Represents a single decision tree structure from a LightGBM model.

    Attributes:
        leaf_values (List[float]): Values of the leaf nodes
        threshold_values (List[float]): Split threshold values
        threshold_ids (List[int]): Index of threshold values
        left_child_ids (List[int]): Index of left child nodes
        right_child_ids (List[int]): Index of right child nodes
        split_gain (List[float]): Gain values for each split
    """
    leaf_values: List[float]
    threshold_values: List[float]
    threshold_ids: List[int]
    left_child_ids: List[int]
    right_child_ids: List[int]
    split_gain: List[float]

    def __init__(self):
        self.leaf_values = []
        self.feature_ids = []
        self.thresholds = []
        self.threshold_ids = []
        self.nodes = []
        self.split_gain = []

@dataclass
class LightGBM:
    """Internal representation of LightGBM models in Python.

    This class handles loading LightGBM models, converting them to C++ code,
    and performing various post-training optimization methods.

    Attributes:
        trees (List[Tree]): List of decision trees in the model
        thresholds (List[float]): Unique threshold values across all trees
        features (int): Number of input features
        num_class (int): Number of output classes (1 for binary/regression)
        objective (str): Model objective ("regression" or "binary")
        quantization_diff (str): Quantization target ("leafs", "thresholds", "both")
        quantization_type (str): Quantization method ("affine" or "scale")
        bits (int): Bit width for quantization (8 or 16)
        alpha (float): Maximum value for scaling quantization
        beta (float): Minimum value for affine quantization
        s (float): Scale factor for quantization
        z (float): Zero point for affine quantization
    """
    trees: List[Tree]
    thresholds: List[float]
    features: int
    num_class: int
    objective: str
    quantization_diff: str
    quantization_type: str
    bits: int
    alpha: float
    beta: float
    s: float
    z: float

    def __init__(self):
        self.trees = []
        self.thresholds = []
        self.features = 0
        self.num_class = 0
        self.bits = None
        self.alpha = 0.0
        self.beta = 0.0
        self.s = 0
        self.z = 0
        self.objective = ""
        self.quantization_diff = ""
        self.quantization_type = ""
        
    def load(self, filename: str) -> Tuple[List[Tree], List[int], List[float], int]:
        """Loads a LightGBM model from a text file.

        Parses the model file and extracts tree structures, thresholds,
        and model parameters. Builds internal representation of the model.

        Args:
            filename (str): Path to the LightGBM text model file

        Raises:
            Exception: If no valid tree splits are found in the model
        """
        current_tree = Tree()
        threshold_dict = {}

        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                tokens = line.split('=', 1)
                header = tokens[0] + '='
                values = tokens[1].split() if len(tokens) > 1 else []

                if '=' in header:
                    header = header[:header.index('=') + 1]

                if header == "leaf_value=":
                    current_tree.leaf_values = [float(v) for v in values]
                elif header == "split_feature=":
                    current_tree.feature_ids = [int(v) for v in values]
                elif header == "threshold=":
                    current_tree.threshold_values = [float(v) for v in values]
                    for v in values:
                        float_v = float(v)
                        if float_v in threshold_dict:
                            threshold_id = threshold_dict[float_v]
                        else:
                            threshold_id = len(threshold_dict)
                            threshold_dict[float_v] = threshold_id
                        current_tree.threshold_ids.append(threshold_id)         
                elif header == "left_child=":
                    current_tree.left_child_ids = [int(v) for v in values]
                elif header == "right_child=":
                    current_tree.right_child_ids = [int(v) for v in values]
                elif header == "split_gain=":
                    current_tree.split_gain = [float(v) for v in values]
                    if not current_tree.split_gain:
                        current_tree.split_gain = [0]
                elif header == "max_feature_idx=":
                    self.features = int(values[0])
                elif header == "num_class=":
                    self.num_class = int(values[0])
                elif header == "objective=":
                    self.objective = values[0]
                elif "shrinkage=" in header:
                    self.trees.append(current_tree)
                    current_tree = Tree()

        self.thresholds = [k for k, _ in sorted(threshold_dict.items(), key=lambda x: x[1])]

        if len(self.thresholds) == 0: 
            raise Exception("Hyperparameters too constrictive: No tree splits produced")
 
    def predict(self, data):
        """Makes predictions for multiple data instances using the loaded model.

        Args:
            data: Array-like of shape (n_samples, n_features)

        Returns:
            List of predictions for each input instance
        """
        predicitons = []
        for i, v in enumerate(data):
            predicitons.append(self.__predict_instance(v))
        return predicitons

    def __predict_instance(self, values):
        """Makes a prediction for a single instance using the loaded model.

        Traverses all trees in the model and combines their predictions according
        to the model objective (binary classification, multiclass, or regression).
        Applies a stable sigmoid function for binary classification and a softmax
        function for multiclass classification.

        Args:
            values: Array-like of features for a single instance

        Returns:
            Predicted value or class
        """
        result = 0
        if self.num_class == 1:
            for i, tree in enumerate(self.trees):
                result += self.__predict_node(tree, 0, values)
            if self.objective != "regression":
                if result >= 0:
                    result = 1 / (1 + np.exp(-result))
                else:
                    stable = np.exp(result)
                    result = stable / (1 + stable)
                result = 1 if result > 0.5 else 0
        else:
            votes = [0 for i in range(self.num_class)]
            for i, tree in enumerate(self.trees):
                votes[i % self.num_class] += self.__predict_node(tree, 0, values)
            exp_votes = np.exp(np.array(votes) - np.max(votes))
            result = exp_votes / exp_votes.sum()
            result = result.argmax()
        return result
                
    def __predict_node(self, tree: Tree, i: int, values):
        """Recursively traverses a tree to make a prediction.

        Args:
            tree: Decision tree to traverse
            i: Current node index
            values: Feature values for prediction

        Returns:
            Prediction value at the reached leaf node
        """
        if len(tree.leaf_values) == 1:
            return tree.leaf_values[0]
        if i < 0:
            return tree.leaf_values[abs(i)-1]
        else:
            if values[tree.feature_ids[i]] <= self.thresholds[tree.threshold_ids[i]]:
                return self.__predict_node(tree, tree.left_child_ids[i], values)
            else:
                return self.__predict_node(tree, tree.right_child_ids[i], values)
    
    def __print_node(self, out_file, tree: Tree, i: int, indent_level: int) -> None:
        """Recursively generates C++ code for a binary/regression tree node.

        Writes the decision tree logic as nested if-else statements for
        binary classification or regression models.

        Args:
            out_file: File object to write C++ code to
            tree: Decision tree being converted
            i: Current node index
            indent_level: Current indentation level for code formatting
        """
        indent = '\t' * indent_level
        if len(tree.leaf_values) == 1:
            out_file.write(f"{indent}result += {tree.leaf_values[0]};\n")
            return
        if i < 0:
            out_file.write(f"{indent}result += {tree.leaf_values[abs(i)-1]};\n")
        else:
            out_file.write(f"{indent}if (values[{tree.feature_ids[i]}] <= thresholds[{tree.threshold_ids[i]}]) {{\n")
            self.__print_node(out_file, tree, tree.left_child_ids[i], indent_level + 1)
            out_file.write(f"{indent}}} else {{\n")
            self.__print_node(out_file, tree, tree.right_child_ids[i], indent_level + 1)
            out_file.write(f"{indent}}}\n")

    def __print_node_multi(self, out_file, tree: Tree, i: int, j: int, indent_level: int) -> None:
        """Recursively generates C++ code for a multiclass tree node.

        Similar to __print_node but handles multiple output classes by
        adding raw scores to each class in a votes array.

        Args:
            out_file: File object to write C++ code to
            tree: Decision tree being converted
            i: Current node index
            j: Current class index
            indent_level: Current indentation level for code formatting
        """
        indent = '\t' * indent_level
        if len(tree.leaf_values) == 1:
            out_file.write(f"{indent}votes[{j}] += {tree.leaf_values[0]};\n")
            return
        if i < 0:
            out_file.write(f"{indent}votes[{j}] += {tree.leaf_values[abs(i)-1]};\n")
        else:
            out_file.write(f"{indent}if (values[{tree.feature_ids[i]}] <= thresholds[{tree.threshold_ids[i]}]) {{\n")
            self.__print_node_multi(out_file, tree, tree.left_child_ids[i], j, indent_level + 1)
            out_file.write(f"{indent}}} else {{\n")
            self.__print_node_multi(out_file, tree, tree.right_child_ids[i], j, indent_level + 1)
            out_file.write(f"{indent}}}\n")

    def generate(self, directory: str, filename: str) -> None:
        """Generates C++ code for the LightGBM model.

        Args:
            directory: Directory to save the generated C++ header file
            filename: Name of the generated C++ header file (without extension)
        """
        leaf_type = "float"
        threshold_type = "float"
        if self.quantization_diff == "both":
            if self.bits == 8:
                leaf_type = "int8_t"
                threshold_type = "int8_t"
            elif self.bits == 16:
                leaf_type = "int16_t"
                threshold_type = "int16_t"
        elif self.quantization_diff == "leafs":
            if self.bits == 8:
                leaf_type = "int8_t"
            elif self.bits == 16:
                leaf_type = "int16_t"
        elif self.quantization_diff == "thresholds":
            if self.bits == 8:
                threshold_type = "int8_t"
            elif self.bits == 16:
                threshold_type = "int16_t"

        os.makedirs(directory, exist_ok=True)
        output_path = os.path.join(directory, f"{filename}.h")
        
        with open(output_path, 'w') as out:
            out.write("#pragma once\n")
            out.write("#include <stdint.h>\n")
            out.write("namespace LightGBM { \n")
            out.write("\t\tclass CovTypeClassifier {\n")
            out.write("\t\tpublic:\n")
            out.write(f"\t\t\tfloat predict(float values[{self.features + 1}]) {{\n")
            #print(self.thresholds)
            out.write(f"\t\t\t\t{threshold_type} thresholds[{len(self.thresholds)}] = {{")

            for i, threshold in enumerate(self.thresholds):
                if i > 0: out.write(", ")
                out.write(f"{threshold}")
            out.write("};\n")

            ### Quantization of feature values if threshold quantization is enabled ###
            if self.quantization_diff == "thresholds" or self.quantization_diff == "both":
                if self.quantization_type == "affine":
                    s, z = self.affineGet(self.beta, self.alpha)
                    out.write(f"\t\t\t\tfor(int i = 0; i < {self.features + 1}; i++) values[i] = constrain(round({s} * values[i] + {z}), {-2**(self.bits-1)}, {2**(self.bits-1)-1}); \n")
                elif self.quantization_type == "scale":
                    s = self.scaleGet(self.alpha)
                    out.write(f"\t\t\t\tfor(int i = 0; i < {self.features + 1}; i++) values[i] = constrain(round({s} * values[i]), {-2**(self.bits-1)+1}, {2**(self.bits-1)-1}); \n")
            
            if self.num_class == 1:
                out.write(f"\t\t\t\t{leaf_type} result = 0;\n")
                ### Iterate over all trees and print the binary tree structure ###
                for i, tree in enumerate(self.trees):
                    out.write(f"\t\t\t\t// tree {i} ...\n")
                    self.__print_node(out, tree, 0, 4)
                if self.objective == "regression":
                    ### Dequantization of leaf value sum if leaf quantization is enabled ###
                    if self.quantization_diff == "leafs" or self.quantization_diff == "both":
                        if self.quantization_type == "affine":
                            out.write(f"\t\t\treturn (1/{self.s}) * (result - {len(self.trees)*self.z});\n")
                        elif self.quantization_type == "scale":
                            out.write(f"\t\t\treturn result/{self.s};\n")
                    else:
                        out.write("\t\t\treturn result;\n")
                else:
                    out.write("\t\t\tresult = 1.0f/(1.0f + exp(-1.0f * result));\n")
                    out.write("\t\t\treturn result > 0.5f ? 1 : 0;\n")
                out.write("\t\t}\n")
                out.write("\t};\n")
                out.write("}\n")    
            else:
                out.write(f"\t\t\t\t{leaf_type} votes[{self.num_class}] = {{ 0{'.0f' if self.bits is None else ''} }};\n")
                ### Iterate over all trees and print the multiclass tree structure ###
                for i, tree in enumerate(self.trees):
                        out.write(f"\t\t\t\t// tree {i} ...\n")
                        self.__print_node_multi(out, tree, 0, i % self.num_class, 4)
                out.write(f"\t\t\tfloat max_val = votes[0];\n")
                out.write(f"\t\t\tfor(int i = 1; i < {self.num_class}; ++i) if(votes[i] > max_val) max_val = votes[i];\n")
                out.write("\t\t\tfloat sum = 0.0f;\n")
                out.write(f"\t\t\tfor(int i = 0; i < {self.num_class}; ++i) sum += exp(votes[i] - max_val);\n")
                out.write("\t\t\tint max_idx = 0;\n")
                out.write(f"\t\t\tfor(int i = 1; i < {self.num_class}; ++i) if(votes[i] > votes[max_idx]) max_idx = i;\n")
                out.write("\t\t\treturn max_idx;\n")
                out.write("\t\t}\n")
                out.write("\t};\n")
                out.write("}\n")

    def estimateMemory(self):
        """Estimates the memory usage of the model.

        Sums the memory usage of all leaf values, threshold values, and
        splits across all trees in the model.
        This is only a rough estimate and thus not used in the experiment.

        Returns:
            int: Estimated memory usage in bytes
        """
        threshold_mem = sum([1 if isinstance(x, int) and self.bits == 8 else 2 if isinstance(x, int) and self.bits == 16 else 4 for x in self.thresholds])
        leaf_mem = 0
        condition_mem = 0
        for tree in self.trees:
            leaf_mem += sum([1 if isinstance(x, int) and self.bits == 8 else 2 if isinstance(x, int) and self.bits == 16 else 4 for x in tree.leaf_values])
            condition_mem += len(tree.feature_ids) * 1 #Each split is a estimated as single byte
        return threshold_mem + leaf_mem + condition_mem
    
    def returnMemory(self):
        """Compiles the model to read the memory usage on an Arduino device.
        This method will only work if the Arduino cli is installed and is in PATH.
        Requires an Arduiono UNO device to be connected or be known to the computer.

        This method was only tested on Windows and may not work on other operating systems.
        (If the method does not work, the memory usage can be estimated using the estimateMemory method)
        
        Requirements:
            Arduino CLI installed and in PATH
            Arduino UNO device connected or known to the computer
            Windows operating system

        Returns:
            int: Program storage size in bytes
        """
        unique_id = uuid.uuid4()
        base_path = os.path.join(tempfile.gettempdir(), "LightGBM-Experiment")
        
        unique_dir = os.path.join(base_path, f"{unique_id}")
        sketch_dir = os.path.join(unique_dir, "sketch")
        os.makedirs(sketch_dir, exist_ok=True)
        
        sketch_path = os.path.join(sketch_dir, "sketch.ino")
        self.generate(sketch_dir, "model")
        liststorage = []
        ### Creates a sketch ino file that uses the generated model header to make three predictions###
        with open(sketch_path, "w") as sketch_file:
            sketch_file.write('#include "model.h"\n\n')
            sketch_file.write("LightGBM::CovTypeClassifier classifier;\n\n")
            sketch_file.write(f"#define INPUT_SIZE {self.features + 1}\n\n")
            sketch_file.write("float input[INPUT_SIZE];\n\n")

            sketch_file.write("void setup() {\n")
            sketch_file.write("\tSerial.begin(9600);\n")
            sketch_file.write("}\n\n")
            sketch_file.write("void fillArray(float * array) {\n");
            sketch_file.write("\tlong randomVal;\n")
            sketch_file.write("\tdouble randomDec;\n")

            sketch_file.write("\tfor(int i = 0; i<INPUT_SIZE; i++){\n")
            sketch_file.write("\t\trandomVal = random(-200000,200001);\n")
            sketch_file.write("\t\trandomDec = (double)randomVal / 10000;\n")
            sketch_file.write("\t\tarray[i] = randomDec;\n")
            sketch_file.write("\t}\n")
            sketch_file.write("\t}\n")
            sketch_file.write("void loop() {\n")
            for i in range(3):
                sketch_file.write(f"\tfillArray(input);\n")

                sketch_file.write(f"\tfloat output{i} = classifier.predict(input);\n")
                sketch_file.write(f"\tSerial.println(output{i});\n")

            sketch_file.write("\tdelay(1000);\n")
            sketch_file.write("}\n")
        try:
            ### Generates a C++ header file for the model ###
            ### Compiles the sketch using the Arduino CLI ###
            #print(sketch_path)
            platforms = ['arduino:avr:uno', 'arduino:avr:leonardo', 'arduino:samd:mkr1000', 'arduino:samd:mkrgsm1400', 'arduino:esp32:unowifi', 'esp32:esp32:adafruit_feather_esp32_v2', 'esp32:esp32:sparklemotion']
            # arduino: avr:leonardo arduino:samd:mkr1000 arduino:samd:mkrgsm1400 arduino:esp32:unowifi adafruit:samd:feather_m0 raspberrypi:pi:pico
            for platform in platforms:
                result = subprocess.run(['/opt/homebrew/bin/arduino-cli', 'compile', '--fqbn', platform, sketch_path],
                                    capture_output=True, text=True)
                ### Error value ###
                program_storage = 999999

                ### Parses the output to get the program storage size ###
                if result.stdout:
                    storage_match = re.search(r'Sketch uses (\d+) bytes', result.stdout)

                    if storage_match:
                        program_storage = int(storage_match.group(1))
                    elif (overflow_match := re.search(r"region `text' overflowed by (\d+) bytes", result.stderr)):
                        print(f"Overflow detected by {int(overflow_match.group(1))} bytes")
                        program_storage = 131072 + int(overflow_match.group(1))
                liststorage.append(program_storage)
            return liststorage

        finally:
            if os.path.exists(unique_dir):
               try:
                   shutil.rmtree(unique_dir)
               except PermissionError as e:
                   print(f"Permission error: {e}")
    
    ### Quantization functions require self.bits to be set to either 8 or 16###
    def setBits(self, bits: int):
        """Sets the bit width for quantization.

        Args:
            bits: Bit width for quantization (8 or 16)
        """
        self.bits = bits

    def setQuant(self, quantization_diff: int):
        """Sets the quantization target.

        Args:
            quantization_diff: Quantization target ("leafs", "thresholds", "both")
        """
        self.quantization_diff = quantization_diff

    def setQuantType(self, quantization_type: str):
        """Sets the quantization method.

        Args:
            quantization_type: Quantization method ("affine" or "scale")
        """
        self.quantization_type = quantization_type

    def affineGet(self, beta, alpha):
        """Calculates the scale and zero point for affine quantization.

        Args:
            beta: Minimum value for affine quantization
            alpha: Maximum value for affine quantization

        Returns:
            Tuple containing the scale factor and zero point
        """
        s = (2**(self.bits-1) - 1) / (alpha - beta) #(2**(self.bits-1) - (-2**(self.bits-1))) / (alpha - beta) #
        z = -np.round(beta * s)-2**(self.bits-1)
        return s, z

    def affineFunction(self, x, s, z):
        """Applies affine quantization to a value.

        Args:
            x: Value to be quantized
            s: Scale factor
            z: Zero point

        Returns:
            Quantized value
        """
        return np.clip(np.round( s * x + z), a_min= -2**(self.bits-1), a_max = 2**(self.bits-1)-1)
       
    def affineQuantLeafs(self):
        """Applies affine quantization to leaf values."""
        values = [v for t in self.trees for v in t.leaf_values]
        alpha = max(values)
        beta = min(values)
        s, z = self.affineGet(beta, alpha)
        self.s, self.z = s, z
        for tree in self.trees:
            tree.leaf_values = list(map(lambda x: int(self.affineFunction(x, s,z)), tree.leaf_values))     

    def affineQuantThresholds(self, data):
        """Applies affine quantization to threshold values.

        Args:
            data: Array-like of shape (n_samples, n_features)

        Returns:
            Quantized data
        """
        dataCopy = data.copy()
        values = [v for t in self.trees for v in t.threshold_values]
        self.alpha = max(values)
        self.beta = min(values)
        s, z = self.affineGet(self.beta, self.alpha)
        self.thresholds.clear()
        for tree in self.trees:
            tree.threshold_values = list(map(lambda x: int(self.affineFunction(x, s,z)), tree.threshold_values))
            tree.threshold_ids.clear()
            for v in tree.threshold_values:
                if v not in self.thresholds:
                    self.thresholds.append(v)
                tree.threshold_ids.append(self.thresholds.index(v))    
        dataCopy = np.vectorize(lambda x: int(self.affineFunction(x, s, z)))(dataCopy)
        return dataCopy

    def affineQuantization(self, data):
        """Applies affine quantization to both leaf and threshold values.

        Args:
            data: Array-like of shape (n_samples, n_features)

        Returns:
            Quantized data
        """
        self.affineQuantLeafs()
        return self.affineQuantThresholds(data)

    def scaleGet(self, alpha):
        """Calculates the scale factor for scale quantization.

        Args:
            alpha: Maximum value for scale quantization

        Returns:
            Scale factor
        """
        return (2**(self.bits-1)-1)/alpha

    def scaleFunction(self, x, s):
        """Applies scale quantization to a value.

        Args:
            x: Value to be quantized
            s: Scale factor

        Returns:
            Quantized value
        """
        return np.clip(np.round(s * x), a_min= -2**(self.bits-1)+1, a_max = 2**(self.bits-1)-1)

    def scaleQuantLeafs(self):
        """Applies scale quantization to leaf values."""
        alpha = max([abs(v) for t in self.trees for v in t.leaf_values])
        s = self.scaleGet(alpha)
        self.s = s
        for tree in self.trees:
            tree.leaf_values = list(map(lambda x: int(self.scaleFunction(x, s)), tree.leaf_values))

    def scaleQuantThresholds(self, data):    
        """Applies scale quantization to threshold values.

        Args:
            data: Array-like of shape (n_samples, n_features)

        Returns:
            Quantized data
        """
        dataCopy = data.copy()
        self.alpha =  max([abs(v) for t in self.trees for v in t.threshold_values])
        s = self.scaleGet(self.alpha)
        self.thresholds.clear()
        for tree in self.trees:
            tree.threshold_values = list(map(lambda x: int(self.scaleFunction(x,  s)), tree.threshold_values))
            tree.threshold_ids.clear()
            for v in tree.threshold_values:
                if v not in self.thresholds:
                    self.thresholds.append(v)
                tree.threshold_ids.append(self.thresholds.index(v))    
        dataCopy = np.vectorize(lambda x: int(self.scaleFunction(x, s)))(dataCopy)
        return dataCopy

    def scaleQuantization(self, data):
        """Applies scale quantization to both leaf and threshold values.

        Args:
            data: Array-like of shape (n_samples, n_features)

        Returns:
            Quantized data
        """
        self.scaleQuantLeafs()
        data = self.scaleQuantThresholds(data)
        return data

    def killTrees(self, threshold: float, quant=True):
        """Removes trees with low split gain according to a threshold.
        With quant=True, the threshold is set with a percentile value.

        Args:
            threshold: Split gain threshold for removing trees
            quant: Whether to use percentile-based threshold
        """
        ### Removes trees with split gain below the threshold ###
        if quant:
            threshold = np.quantile([sum(tree.split_gain) for tree in self.trees], threshold)     
        self.trees = [tree for tree in self.trees if sum(tree.split_gain) >= threshold]
        self.thresholds.clear()
        ### Updates the threshold values and ids for the remaining trees ###
        for tree in self.trees:
            tree.threshold_ids.clear()
            for v in tree.threshold_values:
                if v not in self.thresholds:
                    self.thresholds.append(v)
                tree.threshold_ids.append(self.thresholds.index(v))
    
    def mergeThresholds(self, num_clusters):
        """Merges similar threshold values using k-means clustering.

        Args:
            num_clusters: Number of clusters for k-means
        """
        if num_clusters >= len(self.thresholds):
            return
        kmeans = KMeans(n_clusters=num_clusters, init="k-means++", random_state=42).fit(np.array(self.thresholds).reshape(-1,1))
        clusters = kmeans.predict(np.array(self.thresholds).reshape(-1,1))
        centroids = kmeans.cluster_centers_.flatten().tolist() 
        self.thresholds = centroids
        ### Updates the threshold values and ids for all trees ###
        for tree in self.trees:
            tree.threshold_ids = [clusters[x] for x in tree.threshold_ids]
            tree.threshold_values = [self.thresholds[x] for x in tree.threshold_ids]