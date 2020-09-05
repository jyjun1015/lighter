#include <vector>
#include <algorithm>

using namespace std;

int solution(int n, vector<int> weak, vector<int> dist) {
	int cntWeakSpot = weak.size();
    for (int i = 0; i < cntWeakSpot; ++i)
		weak.push_back(weak[i] + n);
	
	int answer = dist.size() + 1;	// 초기값: (최대 친구 수 + 1)
	for (int i = 0; i < cntWeakSpot; ++i) {	// 시작 위치를 변경해주며 탐색한다.
		do {
			int cntFriend = 1;		// 첫 번째 친구가 i번째 시작 위치에서 이동한다.
			int pos = weak[i] + dist[cntFriend - 1];
			for (int j = i + 1; j < cntWeakSpot + i; ++j) {
				if (pos < weak[j]) {// 이전 사람이 j번째 약점 검사 못했으므로 친구 더 필요함.
					cntFriend++;	// 필요한 친구 수 추가해준다.
					if (cntFriend > dist.size()) break;		// 더 부를 친구 없다면 반복문 탈출.
					pos = weak[j] + dist[cntFriend - 1];	// 방금 부른 친구가 j번째 시작 위치에서 이동한다.
				}
			}
			answer = min(answer, cntFriend);	// 최소 친구 수 갱신한다.
		} while (next_permutation(begin(dist), end(dist)));	// 모든 친구 조합에 대해서 테스트 하기 위해 순열 사용.
	}
	if (answer > dist.size()) return -1;	// [예외]: 현재 인원으로 약점을 모두 검사할 수 없는 경우.
	return answer;
}
