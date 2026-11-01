import XCTest
@testable import ImageSafety
final class StorageTests: XCTestCase {
    func testRawParameterBytesRemainIdentical() throws {
        let data = Data([0, 0, 0, 128, 0, 0, 0, 0])
        XCTAssertEqual(try ImageSafetySharedSafetyParameters.restore(data, at: URL(fileURLWithPath: "/unused"), expectedBytes: data.count), data)
    }
    func testMalformedCompactHeaderFailsClosed() {
        XCTAssertThrowsError(try ImageSafetySharedSafetyParameters.restore(Data("MQREF001".utf8), at: URL(fileURLWithPath: "/unused"), expectedBytes: 100))
    }
    func testInvalidScoreIsRejected() {
        XCTAssertThrowsError(try ImageSafetyPhotoInspectionContract.Score(.nan, domain: .probability))
        XCTAssertThrowsError(try ImageSafetyPhotoInspectionContract.Score(1.1, domain: .probability))
    }
    func testFixedThresholdBoundary() throws {
        XCTAssertEqual(try ImageSafetyPhotoInspectionContract.Score(0.1, domain: .signedUnitMargin).checked(as: .signedUnitMargin), 0.1)
    }
}
