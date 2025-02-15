import re
import json
import os
import sys
from pathlib import Path
import argparse


class Node:
    def __init__(self, feature=None, threshold=None, class_label=None):
        self.feature = feature
        self.threshold = threshold
        self.class_label = class_label
        self.left = None
        self.right = None


def parse_tree(lines):
    root = Node("root", None, None)
    stack = [root]
    current_depth = -1
    line_number = 0

    try:
        for line in lines:
            line_number += 1
            if line_number == 687:
                pass
            if not line.strip():
                continue

            # Calculate depth by counting '|' characters
            depth = line.count("|")  # Subtract 1 since the first line has one '|'
            line = line.strip()

            print(f"Processing line {line_number}: depth={depth}, content={line}")

            # Create node
            node = Node()

            # Parse line content
            if "--- class:" in line:
                node.class_label = line.split("class:")[1].strip()
                print(f"Created leaf node with class: {node.class_label}")
            else:
                feature_match = re.search(r"--- (.*) <= ([\d.-]+)", line)
                if feature_match:
                    node.feature = feature_match.group(1)
                    node.threshold = float(feature_match.group(2))
                    print(
                        f"Created decision node with feature: {node.feature} <= {node.threshold}"
                    )
                else:
                    feature_match = re.search(r"--- (.*) >  ([\d.-]+)", line)
                    if feature_match:
                        node.feature = feature_match.group(1)
                        node.threshold = float(feature_match.group(2))
                        print(
                            f"Created decision node with feature: {node.feature} > {node.threshold}"
                        )

            # Handle tree structure
            if depth == 0:
                print("Setting root node")
                root = node
                stack = [node]
                current_depth = 0
            else:
                # If we're going deeper in the tree
                if depth > current_depth:
                    print(f"Going deeper: {current_depth} -> {depth}")
                    if not stack:
                        raise ValueError(
                            f"Stack is empty at line {line_number} while trying to go deeper"
                        )
                    parent = stack[-1]
                    if parent.left is None:
                        print("Setting left child")
                        parent.left = node
                    else:
                        print("Setting right child")
                        parent.right = node
                    stack.append(node)
                # If we're going back up or staying at the same level
                else:
                    print(f"Going up/same level: {current_depth} -> {depth}")
                    # Pop nodes until we reach the correct depth
                    while len(stack) > depth:
                        popped = stack.pop()
                        print(
                            f"Popped node: {popped.feature if popped.feature else popped.class_label}"
                        )

                    if not stack:
                        raise ValueError(
                            f"Stack became empty while processing line {line_number}"
                        )

                    parent = stack[-1]
                    if parent.left is None:
                        print("Setting left child")
                        parent.left = node
                    else:
                        print("Setting right child")
                        parent.right = node
                    stack.append(node)
                current_depth = depth

        return root

    except Exception as e:
        print(f"\nError while processing line {line_number}:", file=sys.stderr)
        print(f"Line content: {line.strip()}", file=sys.stderr)
        print(f"Current depth: {depth}", file=sys.stderr)
        print(f"Stack size: {len(stack)}", file=sys.stderr)
        print(f"Stack contents:", file=sys.stderr)
        for i, node in enumerate(stack):
            print(
                f"  {i}: {node.feature if node.feature else node.class_label}",
                file=sys.stderr,
            )
        raise


def node_to_dict(node, left=False):
    if node is None:
        return None

    result = {}

    if node.class_label is not None:
        result["name"] = f"class: {node.class_label}"
        result["class"] = node.class_label
    else:
        if node.feature is not None and node.threshold is not None:
            result["name"] = (
                f"{node.feature} {'≤' if left else '>'} {node.threshold:.2f}"
            )
        elif node.feature == "root":
            result["name"] = "root"
        else:
            result["name"] = "Unknown"

    result["children"] = []
    if node.left:
        left_dict = node_to_dict(node.left, left=True)
        if left_dict:
            result["children"].append(left_dict)
    if node.right:
        right_dict = node_to_dict(node.right, left=False)
        if right_dict:
            result["children"].append(right_dict)

    return result


def generate_html(tree_data):
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Decision Tree Visualization</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            .node {
                cursor: pointer;
                margin: 5px;
                padding: 8px 12px;
                border: 1px solid #ccc;
                border-radius: 4px;
                display: inline-block;
                background-color: white;
                transition: all 0.3s ease;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .node:hover {
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                transform: translateY(-1px);
            }
            .feature {
                color: #2196F3;
                font-weight: bold;
            }
            .class {
                color: #4CAF50;
                font-weight: bold;
            }
            .children {
                margin-left: 40px;
                border-left: 2px dashed #ccc;
                padding-left: 20px;
                transition: all 0.3s ease;
            }
            .collapsed > .children {
                display: none;
            }
            .node::before {
                content: '▼';
                margin-right: 8px;
                color: #666;
                display: inline-block;
                transition: transform 0.2s ease;
            }
            .collapsed::before {
                transform: rotate(-90deg);
            }
            .leaf::before {
                content: '●';
                color: #4CAF50;
            }
            .controls {
                margin-bottom: 20px;
                position: sticky;
                top: 10px;
                background-color: rgba(255,255,255,0.9);
                padding: 10px;
                border-radius: 4px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                z-index: 1000;
            }
            button {
                padding: 8px 16px;
                margin-right: 10px;
                border: none;
                border-radius: 4px;
                background-color: #2196F3;
                color: white;
                cursor: pointer;
                transition: background-color 0.2s ease;
                font-weight: bold;
            }
            button:hover {
                background-color: #1976D2;
            }
            .search {
                margin-left: 20px;
                display: inline-block;
            }
            .search input {
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
                width: 200px;
            }
            .dimmed {
                opacity: 0.3;
                transition: opacity 0.3s ease;
            }
        </style>
    </head>
    <body>
        <div class="controls">
            <button onclick="expandAll()">Expand All</button>
            <button onclick="collapseAll()">Collapse All</button>
            <div class="search">
                <input type="text" id="searchInput" placeholder="Search..." onkeyup="searchTree()">
            </div>
        </div>
        <div id="tree"></div>
        <script>
            const treeData = %s;
            
            function createNode(data) {
                const node = document.createElement('div');
                node.className = 'node';
                
                if (data.class) {
                    node.className += ' leaf';
                    node.innerHTML += `<span class="class">${data.class}</span>`;
                } else {
                    node.className += ' expanded';
                    node.innerHTML += `<span class="feature">${data.name}</span>`;
                    
                    const children = document.createElement('div');
                    children.className = 'children';
                    
                    if (data.children) {
                        data.children.forEach(child => {
                            children.appendChild(createNode(child));
                        });
                    }
                    
                    node.appendChild(children);
                    
                    node.addEventListener('click', (e) => {
                        e.stopPropagation();
                        node.classList.toggle('collapsed');
                    });
                }
                
                return node;
            }
            
            function expandAll() {
                document.querySelectorAll('.node').forEach(node => {
                    node.classList.remove('collapsed');
                });
            }
            
            function collapseAll() {
                document.querySelectorAll('.node:not(.leaf)').forEach(node => {
                    node.classList.add('collapsed');
                });
            }
            
            function searchTree() {
                const searchText = document.getElementById('searchInput').value.toLowerCase();
                const nodes = document.querySelectorAll('.node');
                
                if (!searchText) {
                    nodes.forEach(node => {
                        node.classList.remove('dimmed');
                    });
                    return;
                }
                
                nodes.forEach(node => {
                    const text = node.textContent.toLowerCase();
                    if (text.includes(searchText)) {
                        node.classList.remove('dimmed');
                        // Expand parents
                        let parent = node.parentElement;
                        while (parent) {
                            if (parent.classList.contains('children')) {
                                const parentNode = parent.parentElement;
                                parentNode.classList.remove('collapsed');
                                parentNode.classList.remove('dimmed');
                            }
                            parent = parent.parentElement;
                        }
                    } else {
                        node.classList.add('dimmed');
                    }
                });
            }
            
            document.getElementById('tree').appendChild(createNode(treeData));
            
            // Initially collapse all nodes except the root
            collapseAll();
            document.querySelector('.node').classList.remove('collapsed');
        </script>
    </body>
    </html>
    """

    return html_template % json.dumps(tree_data)


def convert_tree_to_html(input_file, output_file):
    try:
        print(f"Reading input file: {input_file}")
        # Read the tree text file
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        print(f"Parsing tree with {len(lines)} lines")
        # Parse the tree
        root = parse_tree(lines)

        print("Converting tree to dictionary")
        # Convert to dictionary format
        tree_data = node_to_dict(root)

        print("Generating HTML")
        # Generate HTML
        html_content = generate_html(tree_data)

        print(f"Writing output file: {output_file}")
        # Write HTML file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        print("Conversion completed successfully!")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        print(f"Error type: {type(e)}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        # Set up argument parsing
        parser = argparse.ArgumentParser(
            description="Convert a decision tree text file to HTML."
        )
        parser.add_argument(
            "input_file", help="Path to the input decision tree text file."
        )
        parser.add_argument("output_file", help="Path to the output HTML file.")
        args = parser.parse_args()  # Parse the command line arguments

        print(f"Input file path: {args.input_file}")
        print(f"Output file path: {args.output_file}")

        if not os.path.exists(args.input_file):
            print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
            sys.exit(1)

        convert_tree_to_html(args.input_file, args.output_file)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        print(f"Error type: {type(e)}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
