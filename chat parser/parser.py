import re
import pandas as pd

def parse_chat_file(file_path):
    trades = []
    trade_pattern = re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+" 
        r"(?P<counterparty>[A-Za-z/ ]+)\s+" 
        r"(?P<direction>Buys|Sells)\s+/\s+" 
        r"(?P<transacting_company>[A-Za-z/ ]+)\s+" 
        r"(?P<quantity>\d+/day)\s+at\s+" 
        r"(?P<price>\d+\.\d{2})"
    )

    with open(file_path, 'r') as f:
        for line in f:
            match = trade_pattern.search(line)
            if match:
                trade_info = match.groupdict()
                trades.append(trade_info)

    return pd.DataFrame(trades)


def main():
    file_path = "C:\\Users\\chena\\Desktop\\code\\backtesterv2\\chat parser\\bloomberg_chat_log.txt" 
    trades_df = parse_chat_file(file_path)
    if not trades_df.empty:
        output_file = "C:\\Users\\chena\\Desktop\\code\\backtesterv2\\chat parser\\parsed_trades.xlsx"
        trades_df.to_excel(output_file, index=False)
        print(f"Parsed trades saved to {output_file}")
    else:
        print("No valid trades found.")

if __name__ == "__main__":
    main()
