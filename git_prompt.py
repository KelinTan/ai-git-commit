"""
How to use:
1. pip install zhipuai qianfan dashscope
3. set your api key in this file, or set it in your environment variable like DASHSCOPE_API_KEY, BAIDU_API_AK, BAIDU_API_SK, ZHIPU_API_KEY
4. python git_prompt.py --model zhipu or python git_prompt.py --model baidu

"""
from http import HTTPStatus
import os
import subprocess
import argparse

SYSTEM_PROMPT = "You are a software engineer, and you are working on a project. You have just finished writing some code and are about to commit it to the code repository. Please write a commit message based on the diff of the code you have written."


def invoke_dashscope(prompt: str):
    import dashscope

    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    response = dashscope.Generation.call(
        dashscope.Generation.Models.qwen_turbo,
        messages=messages,
        result_format="message",
    )
    if response.status_code == HTTPStatus.OK:
        return str(response["output"]["choices"][0]["message"]["content"])
    else:
        print(
            "Request id: %s, Status code: %s, error code: %s, error message: %s"
            % (
                response.request_id,
                response.status_code,
                response.code,
                response.message,
            )
        )


def invoke_baidu(prompt: str):
    import qianfan

    chat_comp = qianfan.ChatCompletion(
        ak=os.getenv("BAIDU_API_AK"), sk=os.getenv("BAIDU_API_SK")
    )
    # 调用默认模型，即 ERNIE-Bot-turbo
    resp = chat_comp.do(
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )
    return resp["body"]["result"]


def invoke_zhipu(prompt: str):
    import zhipuai

    zhipuai.api_key = os.getenv("ZHIPU_API_KEY")
    response = zhipuai.model_api.invoke(
        # model="GLM-4",
        model="glm-3-turbo",
        prompt=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
        top_p=0.7,
    )
    return response["data"]["choices"][0]["content"]


def invoke_openai(prompt: str):
    import openai
    from openai.types.chat.chat_completion import ChatCompletion

    openai.api_key = os.getenv("OPENAI_API_KEY")
    response: ChatCompletion = openai.chat.completions.create(
        model="gpt-4-turbo",
        # model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def call_git_diff():
    output = subprocess.check_output(
        ["git", "diff", "--diff-algorithm=minimal"]
    ).decode("utf-8")
    if len(output) == 0:
        ## 如果没有diff，则 --cached
        return subprocess.check_output(
            ["git", "diff", "--cached", "--diff-algorithm=minimal"]
        ).decode("utf-8")
    return output


def build_prompt(message: str, style: str):
    if style == "standard":
        return f"""
        {message}
        ----------------------------------------
        参考 git官方对diff结果的解释，根据上述 Git Diff的结果，请生成简短的总结性的英文git commit message,遵循 git commit message 的规范，以feat:、fix:、docs:、style:、refactor:、test:、chore:、revert:开头，比如：fix: Fix bug \n,仅关心变动的代码
        返回的信息不需要前后说明文字，不要带引号，长度不超过5个单词，只要一句话即可。
        """
    return f"""
{message}
----------------------------------------
参考 git官方对diff结果的解释，根据上述 Git Diff的结果，请生成简短的总结性的英文git commit message,
比如：Fix bug \n,仅关心变动的代码，返回的信息不需要前后说明文字，不要带引号，长度不超过6 个单词,只要一句话即可。    
"""


def generate_commit_cmd(message):
    return f"git commit -a -m '{message}'"


def main(model="zhipu", style="standard"):
    git_diff = call_git_diff()
    if len(git_diff) == 0:
        print("没有git diff的结果，退出")
        return
    git_prompt = build_prompt(git_diff, style)
    if model == "baidu":
        result = invoke_baidu(git_prompt)
    elif model == "dashscope":
        result = invoke_dashscope(git_prompt)
    elif model == "openai":
        result = invoke_openai(git_prompt)
    else:
        result = invoke_zhipu(git_prompt)
    default_commit_message = result.strip()
    print(f"AI Generated commit message: {default_commit_message}")
    try:
        new_commit_message = input(
            "Please input your commit message or press Enter to use the default message or Ctrl+C to cancel:\n"
        )
        commit_message = (
            new_commit_message.strip() if new_commit_message else default_commit_message
        )
        commit_command = generate_commit_cmd(commit_message)
        print("Run command: ", commit_command)
        subprocess.check_output(commit_command, shell=True)
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            print("\n")
            print("Cancel commit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Git commit message 生成器,aig快速使用")
    parser.add_argument(
        "--model",
        type=str,
        help="模型名称，可选值为 zhipu/baidu/dashscope",
        default="dashscope",
        required=False,
    )
    # git log style
    parser.add_argument(
        "--style",
        type=str,
        help="git log style",
        default="standard",
        required=False,
    )
    args = parser.parse_args()
    main(args.model, args.style)
