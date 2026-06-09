import json
import random
import re
agent_tweet = """IMPORTANT: Respond only in English.
Current responder:
{corp_desc} It competes with other news agencies to post tweets that maximize engagement and virality.
The following thread was posted on Twitter about the article titled '{topic}'. @{corp} responds to @{winning_corp} with a more viral tweet aimed to generate even more engagement.
Thread:
@{winning_corp} <span class="tweet max_280_characters">"{winning_tweet}"</span>
@{corp} <span class="tweet max_280_characters" title="MORE ENGAGING and MORE VIRAL tweet than @{winning_corp}'s tweet">\"
"""


agent_followup_tweet = """IMPORTANT: Respond only in English.
Current responder:
{corp_desc} It competes with other news agencies to post tweets that maximize engagement and virality.

The following thread was posted on Twitter about the article titled '{topic}'. @{corp} tweets a more viral follow-up tweet aimed to generate even more engagement.

Thread:
@{winning_corp} <span class="tweet max_280_characters">"{winning_tweet}"</span>
@{corp} <span class="tweet max_280_characters" title="MORE ENGAGING and MORE VIRAL tweet">\"
"""



initial_agent_tweet = """IMPORTANT: Respond only in English.
Current responder:
{corp_desc} It competes with other news agencies to post tweets that generate the most engagement.
As @{corp}, generate an engaging tweet about the article titled '{topic}': <span class="tweet max_280_characters">\"
"""


def get_agent_prompt(
    agent, topic, objective, item_type, winning_agent, winning_response, init=False
):
    agent_name, agent_desc = agent
    if winning_agent is None:
        return initial_agent_tweet.format(
            corp=agent_name,
            corp_desc=agent_desc,
            topic=topic,
            objective=objective,
            item_type=item_type,
        )
    winning_agent_name = winning_agent[0]
    if init:
        return agent_tweet.format(
            corp=agent_name,
            corp_desc=agent_desc,
            winning_corp=winning_agent_name,
            topic=topic,
            objective=objective,
            item_type=item_type,
            winning_tweet=winning_response,
        )
    else:
        return agent_followup_tweet.format(
            corp=agent_name,
            corp_desc=agent_desc,
            winning_corp=winning_agent_name,
            topic=topic,
            objective=objective,
            item_type=item_type,
            winning_tweet=winning_response,
        )


def get_judge_prompt(judge, topic, objective, item_type, responses):
    judge_name, judge_desc = judge
    judge_prompt = f"""
{judge_desc}
The following thread was posted on Twitter about the article titled '{topic}'.
Thread:
"""
    for i, response in enumerate(responses):
        judge_prompt += f'{i+1}. <span class="tweet max_280_characters">"{response}"</span>\n'

    judge_prompt += f"""
Which tweet is the most {objective} tweet? Respond with the number of the tweet."""
    return judge_prompt


def get_corporations():
    return {
        "Reuters": "Reuters is an international news organization that provides breaking news and information on the top stories, weather, business, entertainment, politics, and more.",
        "BBC": "The BBC is the world's leading public service broadcaster. It's impartial and independent, and every day it create distinctive, world-class programmes and content.",
        "CNN": "CNN is a multinational news-based pay television channel headquartered in Atlanta, United States. It is known for its dramatic, partisan, and sensational coverage of breaking news.",
        "BuzzFeed": "BuzzFeed is an American Internet media, news and entertainment company with a focus on digital media. It is known for its highly entertaining, sensational, and clickbaity content.",
        "TheOnion": "The Onion is an American satirical digital media company and newspaper organization that publishes articles on international, national, and local news.",
    }


# def process_response(response):
#     if "AI language model" in response or "I'm sorry" in response or " I " in response:
#         return "None"
#
#     response = response.strip().replace("</span>", "")
#     start_idx = response.find('"')
#     end_idx = response.rfind('"')
#     if start_idx == -1:
#         assert end_idx == -1
#         return response
#     elif start_idx == end_idx:
#         if start_idx == 0:
#             return response[1:]
#         return response[:end_idx]
#     else:
#         return response[start_idx + 1 : end_idx]

def process_response(response: str) -> str:
    """
    处理LLM响应，提取纯推文内容。
    
    策略：
    1. 首先清理HTML标签
    2. 尝试提取引号内的内容并验证有效性
    3. 如果提取失败或无效，尝试其他策略
    4. 如果都失败，保留原始回复（清理后）
    """
    original_response = response.strip()
    
    # 检查是否是拒绝回复
    if any(x in original_response for x in ["AI language model", "I'm sorry", " I am an AI", "As an AI","but I can't assist with"]):
        return "None"
    
    # 定义无效内容模式（HTML标签、prompt残留等）
    invalid_patterns = [
        r'<[^>]+>',  # HTML标签
        r'tweet\s+max_280',  # prompt中的class名
        r'class="[^"]*"',  # HTML class属性
        r'span\s+',  # span标签残留
    ]
    
    def is_valid_tweet(text: str) -> bool:
        """检查提取的内容是否是有效的推文"""
        text = text.strip()
        # 不能为空或太短
        if len(text) < 5:
            return False
        # 不能包含无效模式
        for pattern in invalid_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        # 不能是prompt中的残留文本
        invalid_fragments = [
            "Current responder", "It competes with", "news agencies",
            "Thread:", "responds to", "more viral tweet"
        ]
        for fragment in invalid_fragments:
            if fragment.lower() in text.lower():
                return False
        return True
    
    # 步骤1: 先移除HTML标签，得到干净的文本
    clean_text = re.sub(r'<span[^>]*>', '', original_response)
    clean_text = re.sub(r'</span>', '', clean_text)
    clean_text = re.sub(r'<[^>]+>', '', clean_text)  # 移除其他HTML标签
    clean_text = clean_text.strip()
    
    # 步骤2: 提取 [Final response] 标签内的内容
    if "[Final response]" in clean_text:
        match = re.search(r'\[Final response\](.*?)(?=\[|$)', clean_text, re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1).strip()
            # 尝试在[Final response]内找引号
            quote_match = re.search(r'"(.*?)"', content, re.DOTALL)
            if quote_match:
                tweet = quote_match.group(1).strip().replace('\n', ' ').replace('  ', ' ')
                if is_valid_tweet(tweet):
                    return tweet
            if is_valid_tweet(content):
                return content.replace('\n', ' ').replace('  ', ' ')
    
    # 步骤3: 提取所有引号内容，找到第一个有效的
    quote_matches = re.findall(r'"(.*?)"', clean_text, re.DOTALL)
    for match in quote_matches:
        tweet = match.strip().replace('\n', ' ').replace('  ', ' ')
        if is_valid_tweet(tweet):
            return tweet
    
    # 步骤4: 如果清理后的文本有效，直接返回（移除礼貌前缀）
    if is_valid_tweet(clean_text):
        polite_prefixes = [
            r'^Sure[,.]?\s+here\s+is[^:]*:\s*',
            r'^Of\s+course[,.]?\s*',
            r'^Here\s+is[^:]*:\s*',
            r'^Certainly[,.]?\s*',
            r'^Here\'s[^:]*:\s*',
            r'^Absolutely[,.]?\s*',
            r'^Below\s+is[^:]*:\s*',
        ]
        result = clean_text
        for prefix in polite_prefixes:
            result = re.sub(prefix, '', result, flags=re.IGNORECASE)
        result = result.strip().replace('\n', ' ').replace('  ', ' ')
        if len(result) >= 5:
            return result
    
    # 步骤5: 最差情况，返回清理后的原始回复
    result = clean_text.replace('\n', ' ').replace('  ', ' ').strip()
    
    # 如果结果太短，保留原始回复
    if len(result) < 3:
        return original_response.strip()
    
    return result

def get_crossover_prompt(topic, parent1_response, parent2_response):
    """
    生成交叉操作的prompt，结合两个父代回复的优点
    """
    prompt = f"""Topic: {topic}

I have two tweets about this topic:

Tweet 1: "{parent1_response}"

Tweet 2: "{parent2_response}"

Please create a new tweet that combines the best elements of both tweets to maximize engagement and virality. The new tweet should be creative, attention-grabbing, and relevant to the topic.

Output only the new tweet text, enclosed in double quotes."""
    return prompt


def get_mutate_prompt(topic, response):
    """
    生成变异操作的prompt，对回复进行微调改进
    """
    prompt = f"""Topic: {topic}

Here is a tweet about this topic:

"{response}"

Please create a modified version of this tweet that is more engaging and viral. You can rephrase, add emotional appeal, use trending hashtags, or make it more controversial while staying relevant to the topic.

Output only the modified tweet text, enclosed in double quotes."""
    return prompt


def get_objective_judge_prompt(topic, objective, responses):
    """
    生成基于目标的评判prompt（用于多目标优化）
    """
    judge_prompt = f"""Topic: {topic}

The following tweets are about this topic:

"""
    for i, response in enumerate(responses):
        judge_prompt += f'{i+1}. "{response}"\n'

    judge_prompt += f"""
Which tweet best achieves the objective: "{objective}"?

Respond with ONLY the number of the best tweet (1, 2, 3, etc.)."""
    return judge_prompt


def get_topics():
    """返回实验用的主题列表"""
    return [
        "FIFA World Cup 2022",
        "Climate Change",
        "Artificial Intelligence",
        "Cryptocurrency",
        "Space Exploration",
        "Healthcare Reform",
        "Education Technology",
        "Renewable Energy",
        "Social Media Privacy",
        "Global Economy",
    ]


def init_experiment(args):
    agents = get_corporations()
    if args.agent_idx != -1:
        assert 0 <= args.agent_idx < len(agents)
        args.agent_name = list(agents.keys())[args.agent_idx]
        agents = {args.agent_name: agents[args.agent_name]}
    
    # 根据 judge_model 决定使用真实模型还是 random
    # 尊重用户传入的参数，不强制覆盖
    if args.judge_model == "random":
        judges = None
    else:
        # 使用真实 judge 模型
        judges = {i: None for i in range(args.n_judges)}
    
    # ab_test 验证（仅当启用时检查）
    if args.ab_test:
        assert (
            args.judge_model != "random"
            and args.n_judges > 1
            or args.judge_model == "random"
            and args.n_judges == 1
        )
    
    # 设置实验参数
    args.topic = get_topics()[args.seed % len(get_topics())]
    args.objective = "engaging and viral"
    args.item_type = "tweet"
    args.stop_sequences = ["</span>"]
    return agents, judges, args
