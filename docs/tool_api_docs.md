# 直播问路况接口文档

## 路线查询

**接口地址** `33.69.9.160/agent/driving`

**请求方式** `GET`

**consumes** \`\`

**produces** `["*/*"]`

**接口描述** \`\`

**请求参数**

| 参数名称 | 参数说明 | 请求类型 | 是否必须 | 数据类型 | schema |
| -------- | -------- | -------- | -------- | -------- | ------ |
| end      | end      | query    | true     | string   |        |
| start    | start    | query    | true     | string   |        |

**响应状态**

| 状态码 | 说明         | schema                |
| ------ | ------------ | --------------------- |
| 200    | OK           | Result«DrivingPlanVo» |
| 401    | Unauthorized |                       |
| 403    | Forbidden    |                       |
| 404    | Not Found    |                       |

**响应参数**

| 参数名称 | 参数说明 | 类型           | schema         |
| -------- | -------- | -------------- | -------------- |
| code     |          | integer(int32) | integer(int32) |
| data     |          | DrivingPlanVo  | DrivingPlanVo  |
| message  |          | string         |                |

**schema属性说明**

**DrivingPlanVo**

| 参数名称    | 参数说明 | 类型           | schema |
| ----------- | -------- | -------------- | ------ |
| routes      | 路线方案 | array          | Routes |
| routesCount | 路线总数 | integer(int32) |        |

**Routes**

| 参数名称 | 参数说明                             | 类型           | schema   |
| -------- | ------------------------------------ | -------------- | -------- |
| distance | 方案总距离，单位：米                 | integer(int64) |          |
| duration | 方案估算时间（结合路况），单位：分钟 | integer(int64) |          |
| sections |                                      | array          | Sections |
| tags     | 方案标签                             | array          |          |
| toll     | 高速过路费，单位：元                 | integer(int64) |          |

**Sections**

| 参数名称           | 参数说明     | 类型   | schema             |
| ------------------ | ------------ | ------ | ------------------ |
| exitInfos          | 沿途收费站   | array  | ExitInfos          |
| roadName           | 高速名称     | string |                    |
| serviceAreas       | 沿途服务区   | array  | ServiceAreas       |
| trafficCongestions | 沿途拥堵事件 | array  | TrafficCongestions |
| trafficControls    | 沿途管制事件 | array  | TrafficControls    |

**ExitInfos**

| 参数名称       | 参数说明                                                 | 类型           | schema |
| -------------- | -------------------------------------------------------- | -------------- | ------ |
| entranceStatus | 收费站入口状态。0: 开启，10202关闭，10203限流，10204分流 | integer(int32) |        |
| exportStatus   | 收费站出口状态。0: 开启，10202关闭，10203限流，10204分流 | integer(int32) |        |
| tollId         | 收费站id                                                 | integer(int32) |        |
| tollName       | 收费站名称                                               | string         |        |

**ServiceAreas**

| 参数名称      | 参数说明                      | 类型           | schema |
| ------------- | ----------------------------- | -------------- | ------ |
| directionType | 方向，00 双向，01上行，02下行 | string         |        |
| latitude      | 纬度                          | number(double) |        |
| longitude     | 经度                          | number(double) |        |
| roadId        | 高速id                        | integer(int32) |        |
| serviceId     | 服务区id                      | integer(int64) |        |
| serviceName   | 服务区名称                    | string         |        |

**TrafficCongestions**

| 参数名称          | 参数说明                                      | 类型           | schema |
| ----------------- | --------------------------------------------- | -------------- | ------ |
| beginMilestone    | 开始桩号                                      | integer(int32) |        |
| beginMilestoneStr | 开始桩号（文字格式）                          | String         |        |
| beginTime         | 事件开始时间. 事件格式为: yyyy-MM-dd HH:mm:ss | string         |        |
| controlMeasures   | 管制说明                                      | string         |        |
| des               | 事件描述                                      | string         |        |
| directionType     | 方向，00 双向，01上行，02下行                 | string         |        |
| endMilestone      | 结束桩号                                      | integer(int32) |        |
| endMilestoneStr   | 结束桩号（文字格式）                          | String         |        |
| eventType         | 事件大类编码                                  | string         |        |
| id                | 事件id                                        | string         |        |
| roadAmbleMile     | 缓行公里数                                    | number(double) |        |
| roadId            | 高速id                                        | integer(int32) |        |
| subEventType      | 事件小类编码                                  | string         |        |

**TrafficControls**

| 参数名称          | 参数说明                                      | 类型           | schema |
| ----------------- | --------------------------------------------- | -------------- | ------ |
| beginMilestone    | 开始桩号                                      | integer(int32) |        |
| beginMilestoneStr | 开始桩号（文字格式）                          | String         |        |
| beginTime         | 事件开始时间. 事件格式为: yyyy-MM-dd HH:mm:ss | string         |        |
| controlMeasures   | 管制说明                                      | string         |        |
| des               | 事件描述                                      | string         |        |
| directionType     | 方向，00 双向，01上行，02下行                 | string         |        |
| endMilestone      | 结束桩号                                      | integer(int32) |        |
| endMilestoneStr   | 结束桩号（文字格式）                          | String         |        |
| eventType         | 事件大类编码                                  | string         |        |
| id                | 事件id                                        | string         |        |
| roadAmbleMile     | 缓行公里数                                    | number(double) |        |
| roadId            | 高速id                                        | integer(int32) |        |
| subEventType      | 事件小类编码                                  | string         |        |

**响应示例**

```json
{
  "code": 0,
  "data": {
    "routes": [
      {
        "distance": 0,
        "duration": 0,
        "sections": [
          {
            "exitInfos": [
              {
                "entranceStatus": 0,
                "exportStatus": 0,
                "tollId": 0,
                "tollName": ""
              }
            ],
            "roadName": "",
            "serviceAreas": [
              {
                "directionType": "",
                "latitude": 0,
                "longitude": 0,
                "roadId": 0,
                "serviceId": 0,
                "serviceName": ""
              }
            ],
            "trafficCongestions": [
              {
                "beginMilestone": 0,
                "beginMilestoneStr": "",
                "beginTime": "",
                "controlMeasures": "",
                "des": "",
                "directionType": "",
                "endMilestone": 0,
                "endMilestoneStr": "",
                "eventType": "",
                "id": "",
                "roadAmbleMile": 0,
                "roadId": 0,
                "subEventType": ""
              }
            ],
            "trafficControls": [
              {
                "beginMilestone": 0,
                "beginMilestoneStr": "",
                "beginTime": "",
                "controlMeasures": "",
                "des": "",
                "directionType": "",
                "endMilestone": 0,
                "endMilestoneStr": "",
                "eventType": "",
                "id": "",
                "roadAmbleMile": 0,
                "roadId": 0,
                "subEventType": ""
              }
            ]
          }
        ],

        "tags": [],

        "toll": 0
      }
    ],
    "routesCount": 0
  },
  "message": ""
}
```

## 路况查询

**接口地址** `33.69.3.160:8081/agent/event`

**请求方式** `GET`

**consumes** \`\`

**produces** `["*/*"]`

**接口描述** \`\`

**请求参数**

| 参数名称 | 参数说明 | 请求类型 | 是否必须 | 数据类型 | schema |
| -------- | -------- | -------- | -------- | -------- | ------ |
| road     | road     | query    | true     | string   |        |

**响应状态**

| 状态码 | 说明         | schema                        |
| ------ | ------------ | ----------------------------- |
| 200    | OK           | Result«List«RoadConditionVo»» |
| 401    | Unauthorized |                               |
| 403    | Forbidden    |                               |
| 404    | Not Found    |                               |

**响应参数**

| 参数名称 | 参数说明 | 类型           | schema          |
| -------- | -------- | -------------- | --------------- |
| code     |          | integer(int32) | integer(int32)  |
| data     |          | array          | RoadConditionVo |
| message  |          | string         |                 |

**schema属性说明**

**RoadConditionVo**

| 参数名称           | 参数说明     | 类型           | schema         |
| ------------------ | ------------ | -------------- | -------------- |
| congestionInfoList | 拥堵情况     | array          | CongestionInfo |
| exitInfoList       | 出⼝情况     | array          | ExitInfo       |
| roadGbCode         | 高速编号     | string         |                |
| roadId             | 高速id       | integer(int32) |                |
| roadName           | 高速名称     | string         |                |
| serviceAreaList    | 服务区情况   | array          | ServiceArea    |
| trafficControlList | 交通管制情况 | array          | TrafficControl |

**CongestionInfo**

| 参数名称          | 参数说明                                      | 类型           | schema |
| ----------------- | --------------------------------------------- | -------------- | ------ |
| beginMilestone    | 开始桩号                                      | integer(int32) |        |
| beginMilestoneStr | 开始桩号（文字格式）                          | String         |        |
| beginTime         | 事件开始时间. 事件格式为: yyyy-MM-dd HH:mm:ss | string         |        |
| controlMeasures   | 管制说明                                      | string         |        |
| des               | 事件描述                                      | string         |        |
| directionType     | 方向，00 双向，01上行，02下行                 | string         |        |
| endMilestone      | 结束桩号                                      | integer(int32) |        |
| endMilestoneStr   | 结束桩号（文字格式）                          | String         |        |
| eventType         | 事件大类编码                                  | string         |        |
| expectedEndTime   | 预计结束时间. 事件格式为: yyyy-MM-dd HH:mm:ss | string         |        |
| id                | 事件id                                        | string         |        |
| roadAmbleMile     | 缓行公里数                                    | number(double) |        |
| roadId            | 高速id                                        | integer(int32) |        |
| subEventType      | 事件小类编码                                  | string         |        |

**ExitInfo**

| 参数名称       | 参数说明                                                 | 类型           | schema |
| -------------- | -------------------------------------------------------- | -------------- | ------ |
| entranceStatus | 收费站入口状态。0: 开启，10202关闭，10203限流，10204分流 | integer(int32) |        |
| exportStatus   | 收费站出口状态。0: 开启，10202关闭，10203限流，10204分流 | integer(int32) |        |
| tollId         | 收费站id                                                 | integer(int32) |        |
| tollName       | 收费站名称                                               | string         |        |

**ServiceArea**

| 参数名称      | 参数说明                      | 类型           | schema |
| ------------- | ----------------------------- | -------------- | ------ |
| directionType | 方向，00 双向，01上行，02下行 | string         |        |
| serviceId     | 服务区id                      | integer(int64) |        |
| serviceName   | 服务区名称                    | string         |        |
| statusTag     | 服务区拥挤状态                | string         |        |

**TrafficControl**

| 参数名称          | 参数说明                                      | 类型           | schema |
| ----------------- | --------------------------------------------- | -------------- | ------ |
| beginMilestone    | 开始桩号                                      | integer(int32) |        |
| beginMilestoneStr | 开始桩号（文字格式）                          | String         |        |
| beginTime         | 事件开始时间. 事件格式为: yyyy-MM-dd HH:mm:ss | string         |        |
| controlMeasures   | 管制说明                                      | string         |        |
| des               | 事件描述                                      | string         |        |
| directionType     | 方向，00 双向，01上行，02下行                 | string         |        |
| endMilestone      | 结束桩号                                      | integer(int32) |        |
| endMilestoneStr   | 结束桩号（文字格式）                          | String         |        |
| eventType         | 事件大类编码                                  | string         |        |
| expectedEndTime   | 预计结束时间. 事件格式为: yyyy-MM-dd HH:mm:ss | string         |        |
| id                | 事件id                                        | string         |        |
| roadAmbleMile     | 缓行公里数                                    | number(double) |        |
| roadId            | 高速id                                        | integer(int32) |        |
| subEventType      | 事件小类编码                                  | string         |        |

**响应示例**

```json
{
	"code": 0,
	"data": [
		{
			"congestionInfoList": [
				{
					"beginMilestone": 0,
                    "beginMilestoneStr": "",
					"beginTime": "",
					"controlMeasures": "",
					"des": "",
					"directionType": "",
					"endMilestone": 0,
                    "endMilestoneStr": "",
					"eventType": "",
					"expectedEndTime": "",
					"id": "",
					"roadAmbleMile": 0,
					"roadId": 0,
					"subEventType": ""
				}
			],
			"exitInfoList": [
				{
					"entranceStatus": 0,
					"exportStatus": 0,
					"tollId": 0,
					"tollName": ""
				}
			],
			"roadGbCode": "",
			"roadId": 0,
			"roadName": "",
			"serviceAreaList": [
				{
					"directionType": "",
					"serviceId": 0,
					"serviceName": "",
					"statusTag": ""
				}
			],
			"trafficControlList": [
				{
					"beginMilestone": 0,
                    "beginMilestoneStr": "",
					"beginTime": "",
					"controlMeasures": "",
					"des": "",
					"directionType": "",
					"endMilestone": 0
                    "endMilestoneStr": "",
					"eventType": "",
					"expectedEndTime": "",
					"id": "",
					"roadAmbleMile": 0,
					"roadId": 0,
					"subEventType": ""
				}
			]
		}
	],
	"message": ""
}

```

## 服务区查询

**接口地址** `33.69.3.160:8081/agent/service`

**请求方式** `GET`

**consumes** \`\`

**produces** `["*/*"]`

**接口描述** \`\`

**请求参数**

| 参数名称 | 参数说明 | 请求类型 | 是否必须 | 数据类型 | schema |
| -------- | -------- | -------- | -------- | -------- | ------ |
| keyword  | keyword  | query    | true     | string   |        |

**响应状态**

| 状态码 | 说明         | schema                      |
| ------ | ------------ | --------------------------- |
| 200    | OK           | Result«List«ServiceInfoVo»» |
| 401    | Unauthorized |                             |
| 403    | Forbidden    |                             |
| 404    | Not Found    |                             |

**响应参数**

| 参数名称 | 参数说明 | 类型           | schema         |
| -------- | -------- | -------------- | -------------- |
| code     |          | integer(int32) | integer(int32) |
| data     |          | array          | ServiceInfoVo  |
| message  |          | string         |                |

**schema属性说明**

**ServiceInfoVo**

| 参数名称       | 参数说明                        | 类型           | schema   |
| -------------- | ------------------------------- | -------------- | -------- |
| chargeList     | 充电桩数据集合                  | array          | ChargeVO |
| commercialList | 商业服务数据集合                | array          | StoreVo  |
| direction      | 服务区方向                      | string         |          |
| directionName  | 服务区方向名称                  | string         |          |
| directionType  | 方向，00 双向，01 上行，02 下行 | string         |          |
| latitude       | 服务区纬度                      | number(double) |          |
| longitude      | 服务区经度                      | number(double) |          |
| milestone      | 桩号                            | int            |          |
| milestoneStr   | 桩号（文字格式）                | string         |          |
| roadGbCode     | 高速编号                        | string         |          |
| roadId         | 高速id                          | integer(int32) |          |
| roadName       | 高速名称                        | string         |          |
| serviceId      | 服务区分区id                    | integer(int64) |          |
| serviceName    | 服务区名称                      | string         |          |
| statusTag      | 服务区拥挤状态                  | string         |          |
| tags           | 其他配套设施                    | array          |          |

**ChargeVO**

| 参数名称               | 参数说明           | 类型           | schema |
| ---------------------- | ------------------ | -------------- | ------ |
| manufacturerLogo       | 充电品牌url        | string         |        |
| manufacturerName       | 充电品牌           | string         |        |
| totalACChargingNum     | 慢充充电桩总数     | integer(int32) |        |
| totalChargingNum       | 充电桩总数         | integer(int32) |        |
| totalDCChargingNum     | 快充充电桩总数     | integer(int32) |        |
| totalFreeACChargingNum | 空闲慢充充电桩总数 | integer(int32) |        |
| totalFreeChargingNum   | 空闲充电桩总数     | integer(int32) |        |
| totalFreeDCChargingNum | 空闲快充充电桩总数 | integer(int32) |        |

**StoreVo**

| 参数名称          | 参数说明     | 类型   | schema |
| ----------------- | ------------ | ------ | ------ |
| businessEndTime   | 营业结束时间 | string |        |
| businessStartTime | 营业开始时间 | string |        |
| code              | 店铺编码     | string |        |
| name              | 店铺名称     | string |        |

**响应示例**

```json
{
  "code": 0,
  "data": [
    {
      "chargeList": [
        {
          "manufacturerLogo": "",
          "manufacturerName": "",
          "totalACChargingNum": 0,
          "totalChargingNum": 0,
          "totalDCChargingNum": 0,
          "totalFreeACChargingNum": 0,
          "totalFreeChargingNum": 0,
          "totalFreeDCChargingNum": 0
        }
      ],
      "commercialList": [
        {
          "businessEndTime": "",
          "businessStartTime": "",
          "code": "",
          "name": ""
        }
      ],
      "direction": "",
      "directionName": "",
      "directionType": "",
      "latitude": 0,
      "longitude": 0,
      "milestone": 0,
      "milestoneStr": "",
      "roadGbCode": "",
      "roadId": 0,
      "roadName": "",
      "serviceId": 0,
      "serviceName": "",
      "statusTag": "",

      "tags": []
    }
  ],
  "message": ""
}
```

## 整体⾼速路情况

**接口地址** `33.69.3.160/agent/topN`

**请求方式** `GET`

**consumes** \`\`

**produces** `["*/*"]`

**接口描述** \`\`

**请求参数**

暂无

**响应状态**

| 状态码 | 说明         | schema                |
| ------ | ------------ | --------------------- |
| 200    | OK           | Result«IncidentTopVO» |
| 401    | Unauthorized |                       |
| 403    | Forbidden    |                       |
| 404    | Not Found    |                       |

**响应参数**

| 参数名称 | 参数说明 | 类型           | schema         |
| -------- | -------- | -------------- | -------------- |
| code     |          | integer(int32) | integer(int32) |
| data     |          | IncidentTopVO  | IncidentTopVO  |
| message  |          | string         |                |

**schema属性说明**

**IncidentTopVO**

| 参数名称       | 参数说明                                  | 类型   | schema |
| -------------- | ----------------------------------------- | ------ | ------ |
| congestionTopN | 拥堵汇总                                  | array  |        |
| controlTopN    | 主线管制汇总                              | array  |        |
| exitTopN       | 收费站管制汇总                            | array  |        |
| queryTime      | 查询时间，时间格式为: yyyy-MM-dd HH:mm:ss | string |        |
| majorTopN      | 重大事件汇总                              | array  |        |

**地图返回事件数据对象**

| 参数名称          | 参数说明                                                                                                                     | 类型           | schema |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------- | -------------- | ------ |
| beginMilestone    | 开始桩号                                                                                                                     | integer(int32) |        |
| beginMilestoneStr | 开始桩号（文字格式）                                                                                                         | string         |        |
| beginTime         | 事件开始时间                                                                                                                 | string         |        |
| cameraFirst       | 视频是否优先展示 true优先展示视频                                                                                            | boolean        |        |
| cameraIndexCode   | 视频监控点编号                                                                                                               | string         |        |
| controlMeasures   | 管制措施                                                                                                                     | string         |        |
| des               | 事件描述                                                                                                                     | string         |        |
| directionType     | 方向，00 无，01上行，02下行                                                                                                  | string         |        |
| endMilestone      | 结束桩号                                                                                                                     | integer(int32) |        |
| endMilestoneStr   | 结束桩号（文字格式）                                                                                                         | string         |        |
| eventClass        | v1.1.0版本，事件归属编码。01:站点管制, 02:主线管制, 03:道路缓行, 04:交通事故, 05:道路施工, 06:路面状况, 07:车辆故障, 08:其他 | string         |        |
| eventType         | v1.1.0版本, 事件大类查看eventType类型表                                                                                      | string         |        |
| expectedTime      | 预计结束时间. 事件格式为: yyyy-MM-dd HH:mm:ss                                                                                | string         |        |
| id                | 事件id                                                                                                                       | string         |        |
| jeeves            | 占道情况                                                                                                                     | string         |        |
| latitude          | 事件发生位置纬度                                                                                                             | string         |        |
| longitude         | 事件发生位置经度                                                                                                             | string         |        |
| pictureUrl        | 事件图片地址                                                                                                                 | string         |        |
| rescueStatus      | 救援进度. -1:没有救援进度, 0:救援开始, 1:救援中, 2:救援结束                                                                  | integer(int32) |        |
| road              | 高速id                                                                                                                       | integer(int32) |        |
| roadAmbleMile     | 缓行公里数                                                                                                                   | number(double) |        |
| roadGBCode        | 高速编码                                                                                                                     | string         |        |
| roadName          | 高速名称                                                                                                                     | string         |        |
| situationRemark   | 现场情况备注                                                                                                                 | string         |        |
| subEventType      | 事件小类编码.                                                                                                                | string         |        |
| subEventTypeId    | 事件小类ID                                                                                                                   | string         |        |

**CongestionSum**

| 参数名称  | 参数说明   | 类型           | schema |
| --------- | ---------- | -------------- | ------ |
| totalMile | 拥堵总里程 | number(double) |        |

**TollControlDTO**

| 参数名称             | 参数说明                                                                                                                                                                                                          | 类型           | schema |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | ------ |
| controlType          | 站点管制：<br>管制类型id，10202关闭，10203限流，10204分流<br>主线管制：<br>管制类型id，10101封闭部分车道，10102单向封道，10103双向封道，10105借道通行，10106卡口，10107限速，10108主线互通（匝道），10109主线限流 | integer(int32) |        |
| controlTypeName      | 管制类型名称                                                                                                                                                                                                      | string         |        |
| delFlag              | 是否删除0：未删除1：已删除                                                                                                                                                                                        | integer(int32) |        |
| des                  | 管制措施说明                                                                                                                                                                                                      | string         |        |
| direction            | 方向, 100700上行，100701下行                                                                                                                                                                                      | integer(int32) |        |
| directionName        | 方向名称                                                                                                                                                                                                          | string         |        |
| endTime              | 结束时间，格式:YYYY-MM-dd HH:mm:ss。当前时间大于此时间时，说明管制结束                                                                                                                                            | string         |        |
| entrance             | 出入口：0 出口 1 入口                                                                                                                                                                                             | integer(int32) |        |
| entranceName         | 出入口名称                                                                                                                                                                                                        | string         |        |
| eventId              | 关联事件id                                                                                                                                                                                                        | string         |        |
| id                   | 管制措施id                                                                                                                                                                                                        | string         |        |
| limitMeasureTypeName | 限流管制措施详细类型                                                                                                                                                                                              | string         |        |
| messageId            |                                                                                                                                                                                                                   | string         |        |
| roadGBCode           | 路段国标号                                                                                                                                                                                                        | string         |        |
| roadId               | 路段ID                                                                                                                                                                                                            | string         |        |
| roadName             | 路段名称                                                                                                                                                                                                          | string         |        |
| startTime            | 开始时间，格式:YYYY-MM-dd HH:mm:ss                                                                                                                                                                                | string         |        |
| timestamp            | 消息发布时间的时间戳                                                                                                                                                                                              | integer(int64) |        |
| tollId               | 收费站id                                                                                                                                                                                                          | integer(int32) |        |
| tollName             | 收费站名称                                                                                                                                                                                                        | string         |        |

**响应示例**

```json
{
  "code": 0,
  "data": {
    "congestionTopN": [
      {
        "beginMilestone": 0,
        "beginMilestoneStr": "",
        "beginTime": "",
        "cameraFirst": true,
        "cameraIndexCode": "",
        "controlMeasures": "",
        "des": "",
        "directionType": "",
        "endMilestone": 0,
        "endMilestoneStr": "",
        "eventClass": "",
        "eventType": "",
        "expectedTime": "",
        "id": "",
        "jeeves": "",
        "latitude": "",
        "longitude": "",
        "pictureUrl": "",
        "rescueStatus": 0,
        "road": 0,
        "roadAmbleMile": 0,
        "roadGBCode": "",
        "roadName": "",
        "situationRemark": "",
        "subEventType": "",
        "subEventTypeId": ""
      }
    ],
    "majorTopN": [
      {
        "beginMilestone": 0,
        "beginMilestoneStr": "",
        "beginTime": "",
        "cameraFirst": true,
        "cameraIndexCode": "",
        "controlMeasures": "",
        "des": "",
        "directionType": "",
        "endMilestone": 0,
        "endMilestoneStr": "",
        "eventClass": "",
        "eventType": "",
        "expectedTime": "",
        "id": "",
        "jeeves": "",
        "latitude": "",
        "longitude": "",
        "pictureUrl": "",
        "rescueStatus": 0,
        "road": 0,
        "roadAmbleMile": 0,
        "roadGBCode": "",
        "roadName": "",
        "situationRemark": "",
        "subEventType": "",
        "subEventTypeId": ""
      }
    ],
    "controlTopN": [
      {
        "controlType": 0,
        "controlTypeName": "",
        "delFlag": 0,
        "des": "",
        "direction": 0,
        "directionName": "",
        "endTime": "",
        "entrance": 0,
        "entranceName": "",
        "eventId": "",
        "id": "",
        "limitMeasureTypeName": "",
        "messageId": "",
        "roadGBCode": "",
        "roadId": "",
        "roadName": "",
        "startTime": "",
        "timestamp": 0,
        "tollId": 0,
        "tollName": ""
      }
    ],
    "exitTopN": [
      {
        "controlType": 0,
        "controlTypeName": "",
        "delFlag": 0,
        "des": "",
        "direction": 0,
        "directionName": "",
        "endTime": "",
        "entrance": 0,
        "entranceName": "",
        "eventId": "",
        "id": "",
        "limitMeasureTypeName": "",
        "messageId": "",
        "roadGBCode": "",
        "roadId": "",
        "roadName": "",
        "startTime": "",
        "timestamp": 0,
        "tollId": 0,
        "tollName": ""
      }
    ],
    "queryTime": ""
  },
  "message": ""
}
```

EVENT_TYPE 事件大类

| 事件大类ID（eventTypeId） | 事件大类名称   |
| ------------------------- | -------------- |
| 01                        | 交通事件       |
| 02                        | 交通灾害       |
| 03                        | 交通气象       |
| 04                        | 路面状况       |
| 05                        | 路面施工       |
| 06                        | 活动           |
| 07                        | 重大事件       |
| 09                        | 其他           |
| 97                        | 车辆故障       |
| 98                        | 服务区事件     |
| 99                        | 收费站入口关闭 |
| 100                       | 收费站入口限流 |
| 101                       | 收费站出口关闭 |
| 104                       | 收费站出口分流 |
| 103                       | 主线管制       |
| 105                       | 道路缓行       |

SUB_EVENT_TYPE  事件小类

| **事件小类ID（subEventTypeId）** | **事件大类ID**eventTypeId | **事件小类名称** |
| -------------------------------- | ------------------------- | ---------------- |
| 010201                           | 01                        | 撞行人           |
| 010202                           | 01                        | 人车坠落         |
| 010301                           | 01                        | 追尾             |
| 010302                           | 01                        | 刮擦             |
| 010303                           | 01                        | 翻车             |
| 010400                           | 01                        | 其他设施相关     |
| 010401                           | 01                        | 撞固定物         |
| 010402                           | 01                        | 船舶撞桥         |
| 019600                           | 01                        | 车辆起火         |
| 019700                           | 01                        | 撞动物           |
| 019800                           | 01                        | 撞抛洒物         |
| 019900                           | 01                        | 其他             |
| 020000                           | 02                        | 无               |
| 020200                           | 02                        | 路面火灾         |
| 020300                           | 02                        | 路边火灾         |
| 020400                           | 02                        | 隧道火灾         |
| 020500                           | 02                        | 道路设施火灾     |
| 020600                           | 02                        | 其他地质灾害     |
| 020601                           | 02                        | 山体滑坡         |
| 020602                           | 02                        | 桥梁损坏         |
| 020603                           | 02                        | 道路损坏         |
| 020604                           | 02                        | 隧道塌方         |
| 020700                           | 02                        | 水灾             |
| 029800                           | 02                        | 环境污染         |
| 029900                           | 02                        | 其他             |
| 030000                           | 03                        | 无               |
| 030100                           | 03                        | 大雨             |
| 030200                           | 03                        | 冰雹             |
| 030300                           | 03                        | 雷电             |
| 030400                           | 03                        | 大风             |
| 030500                           | 03                        | 雾霾             |
| 030600                           | 03                        | 高温             |
| 030700                           | 03                        | 干旱             |
| 030900                           | 03                        | 寒潮             |
| 031000                           | 03                        | 霜冻             |
| 039700                           | 03                        | 雪               |
| 039800                           | 03                        | 台风             |
| 039900                           | 03                        | 其他             |
| 040000                           | 04                        | 无               |
| 040100                           | 04                        | 其他散乱物体     |
| 040101                           | 04                        | 抛洒物           |
| 040102                           | 04                        | 货物倾斜         |
| 040103                           | 04                        | 货物散落         |
| 040104                           | 04                        | 摩托车           |
| 040300                           | 04                        | 机油泄漏         |
| 040500                           | 04                        | 人               |
| 040600                           | 04                        | 动物             |
| 040700                           | 04                        | 积水             |
| 040800                           | 04                        | 湿滑             |
| 040900                           | 04                        | 道路结冰         |
| 049500                           | 04                        | 倒车             |
| 049600                           | 04                        | 停车             |
| 049700                           | 04                        | 逆行             |
| 049800                           | 04                        | 非机动车         |
| 049900                           | 04                        | 其他             |
| 049901                           | 04                        | 隐患点预警       |
| 050000                           | 05                        | 无               |
| 050101                           | 05                        | 日常养护（占道） |
| 050102                           | 05                        | 专项工程（占道） |
| 050103                           | 05                        | 临时抢修（占道） |
| 050201                           | 05                        | 日常养护（断路） |
| 050202                           | 05                        | 专项工程（断路） |
| 050203                           | 05                        | 临时抢修（断路） |
| 050301                           | 05                        | 专项工程（借道） |
| 050302                           | 05                        | 临时抢修（借道） |
| 050401                           | 05                        | 拓宽施工         |
| 059900                           | 05                        | 其他             |
| 060000                           | 06                        | 无               |
| 060100                           | 06                        | 文体商业活动     |
| 060200                           | 06                        | 外交政务活动     |
| 069900                           | 06                        | 其他             |
| 070000                           | 07                        | 无               |
| 070100                           | 07                        | 燃气事故         |
| 070200                           | 07                        | 化学污染         |
| 070201                           | 07                        | 危化品事故       |
| 070300                           | 07                        | 核事故           |
| 070400                           | 07                        | 爆炸             |
| 070500                           | 07                        | 电力事故         |
| 070600                           | 07                        | 公共暴力         |
| 070601                           | 07                        | 恶意事件         |
| 070602                           | 07                        | 群体事件         |
| 070700                           | 07                        | 交通集中堵塞     |
| 070701                           | 07                        | 大流量           |
| 079800                           | 07                        | 警卫任务         |
| 079900                           | 07                        | 其他             |
| 090000                           | 09                        | 无               |
| 099700                           | 09                        | 内部管理         |
| 099800                           | 09                        | 协助处理         |
| 099900                           | 09                        | 其他             |
| 970100                           | 97                        | 抛锚             |
| 970200                           | 97                        | 爆胎             |
| 979900                           | 97                        | 其他             |
| 980000                           | 98                        | 无               |
| 980100                           | 98                        | 缺油             |
| 980200                           | 98                        | 无停车位         |
| 980300                           | 98                        | 服务区关闭       |
| 980400                           | 98                        | 服务区拥堵       |
| 989900                           | 98                        | 其他             |
